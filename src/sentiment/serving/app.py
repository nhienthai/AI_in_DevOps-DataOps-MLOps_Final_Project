"""FastAPI application factory for the sentiment serving service."""

import asyncio
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentiment.config import Settings, get_settings
from sentiment.serving.errors import APIError, install_error_handlers
from sentiment.serving.metrics import (
    BATCH_SIZE,
    DRIFT_PSI,
    INPUT_LENGTH,
    INPUT_TRUNCATIONS,
    LOW_CONFIDENCE,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    DriftReference,
)
from sentiment.serving.middleware import MetricsMiddleware, RequestBodyLimitMiddleware
from sentiment.serving.predictor import Explainer, Prediction, Predictor, StubPredictor
from sentiment.serving.runtime import (
    InferenceOverloadedError,
    InferenceRuntime,
    InferenceTimeoutError,
    ModelNotReadyError,
    PredictorFactory,
)
from sentiment.serving.schemas import (
    Attribution,
    BatchItem,
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    ReloadResponse,
)

_BOOTSTRAP_REFERENCE = DriftReference(
    length_bin_edges=(0.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 100_000.0),
    length_bin_freqs=(0.1, 0.2, 0.3, 0.2, 0.15, 0.05),
    positive_prior=0.5,
)


def _default_predictor_factory(settings: Settings) -> Predictor:
    """Load the configured predictor backend."""
    if settings.predictor_backend == "stub":
        return StubPredictor(max_chars=settings.max_text_length)

    from sentiment.models.registry import load_production_predictor

    return load_production_predictor(
        tracking_uri=settings.mlflow_tracking_uri,
        stage=settings.model_stage,
    )


def _model_version(runtime: InferenceRuntime) -> str:
    """Return a bounded metric label when no model is available."""
    return str(getattr(runtime.predictor, "version", "unloaded"))


def _validate_text(text: str, settings: Settings, version: str) -> None:
    """Apply the same input policy to single and batch inference."""
    if not text.strip():
        PREDICTION_ERRORS.labels(error_type="empty_text", model_version=version).inc()
        raise APIError(422, "empty_text", "Text must not be blank.")
    if len(text) > settings.max_text_length:
        PREDICTION_ERRORS.labels(error_type="text_too_long", model_version=version).inc()
        raise APIError(
            413,
            "text_too_long",
            f"Text exceeds {settings.max_text_length} characters.",
        )


async def _run_predictions(
    runtime: InferenceRuntime, texts: list[str]
) -> tuple[list[Prediction], float]:
    """Translate runtime failures into the stable HTTP error contract."""
    version = _model_version(runtime)
    try:
        predictions, elapsed, _queue_seconds = await runtime.predict(texts)
    except ModelNotReadyError as exc:
        PREDICTION_ERRORS.labels(error_type="model_not_ready", model_version=version).inc()
        raise APIError(503, "model_not_ready", str(exc)) from exc
    except InferenceOverloadedError as exc:
        PREDICTION_ERRORS.labels(error_type="overloaded", model_version=version).inc()
        raise APIError(429, "inference_overloaded", str(exc)) from exc
    except InferenceTimeoutError as exc:
        PREDICTION_ERRORS.labels(error_type="timeout", model_version=version).inc()
        raise APIError(504, "inference_timeout", str(exc)) from exc
    except Exception as exc:
        PREDICTION_ERRORS.labels(error_type="inference_error", model_version=version).inc()
        raise APIError(500, "prediction_failed", "Prediction failed.") from exc
    PREDICTION_LATENCY.labels(model_version=version).observe(elapsed)
    return predictions, elapsed * 1_000.0


def _instrument_prediction(runtime: InferenceRuntime, text: str, prediction: Prediction) -> None:
    version = _model_version(runtime)
    PREDICTION_COUNT.labels(label=prediction.label, model_version=version).inc()
    PREDICTION_CONFIDENCE.observe(prediction.confidence)
    INPUT_LENGTH.observe(len(text))
    if prediction.truncated:
        INPUT_TRUNCATIONS.inc()
    if prediction.confidence < runtime.settings.low_confidence_threshold:
        LOW_CONFIDENCE.inc()
    if runtime.drift is not None:
        runtime.drift.observe(len(text))
        DRIFT_PSI.set(runtime.drift.psi())


def _prediction_response(
    runtime: InferenceRuntime, prediction: Prediction, latency_ms: float
) -> PredictResponse:
    return PredictResponse(
        label=prediction.label,
        score=prediction.score,
        confidence=prediction.confidence,
        model_version=_model_version(runtime),
        truncated=prediction.truncated,
        latency_ms=round(latency_ms, 3),
    )


def create_app(
    predictor_factory: PredictorFactory | None = None,
    explainer: Explainer | None = None,
) -> FastAPI:
    """Build an isolated application instance for Uvicorn or tests."""
    settings = get_settings()
    load_predictor = predictor_factory or _default_predictor_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = InferenceRuntime(settings, load_predictor, _BOOTSTRAP_REFERENCE)
        app.state.runtime = runtime
        app.state.explainer = explainer
        await runtime.start()
        yield
        await runtime.close()

    app = FastAPI(
        title="Sentiment Service",
        version="0.2.0",
        description="Production-oriented, three-class sentiment inference service.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=settings.max_request_body_bytes)
    app.add_middleware(MetricsMiddleware)
    install_error_handlers(app)

    error_responses: dict[int | str, dict[str, Any]] = {
        413: {"model": ErrorResponse, "description": "Configured size limit exceeded."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        429: {"model": ErrorResponse, "description": "Inference capacity is full."},
        500: {"model": ErrorResponse, "description": "Prediction failed."},
        503: {"model": ErrorResponse, "description": "Model is not ready."},
        504: {"model": ErrorResponse, "description": "Prediction timed out."},
    }

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/ready",
        response_model=ReadyResponse,
        tags=["operations"],
        responses={503: error_responses[503]},
    )
    async def ready(request: Request) -> ReadyResponse:
        runtime: InferenceRuntime = request.app.state.runtime
        if runtime.predictor is None:
            raise APIError(503, "model_not_ready", "No model is loaded.")
        return ReadyResponse(status="ready", model_version=runtime.predictor.version)

    @app.post(
        "/reload",
        response_model=ReloadResponse,
        tags=["operations"],
        responses={401: {"model": ErrorResponse}, 503: error_responses[503]},
    )
    async def reload_model(request: Request) -> ReloadResponse:
        """Warm and atomically swap a registry model while preserving the active one."""
        if settings.reload_token is None:
            raise APIError(503, "reload_disabled", "Model reload is not configured.")
        supplied = request.headers.get("x-reload-token", "")
        if not secrets.compare_digest(supplied, settings.reload_token):
            raise APIError(401, "reload_unauthorized", "A valid reload token is required.")
        runtime: InferenceRuntime = request.app.state.runtime
        if not await runtime.reload():
            raise APIError(503, "reload_failed", "The replacement model could not be loaded.")
        return ReloadResponse(status="reloaded", model_version=_model_version(runtime))

    @app.get("/metrics", tags=["operations"], include_in_schema=True)
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    router = APIRouter(prefix="/api/v1", tags=["inference"])

    @router.post("/predict", response_model=PredictResponse, responses=error_responses)
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        runtime: InferenceRuntime = request.app.state.runtime
        _validate_text(payload.text, settings, _model_version(runtime))
        predictions, latency_ms = await _run_predictions(runtime, [payload.text])
        prediction = predictions[0]
        _instrument_prediction(runtime, payload.text, prediction)
        return _prediction_response(runtime, prediction, latency_ms)

    @router.post("/predict/batch", response_model=BatchResponse, responses=error_responses)
    async def predict_batch(request: Request, payload: BatchRequest) -> BatchResponse:
        """Validate independently, then infer all valid items in one model call."""
        runtime: InferenceRuntime = request.app.state.runtime
        version = _model_version(runtime)
        if len(payload.texts) > settings.max_batch_size:
            PREDICTION_ERRORS.labels(error_type="batch_too_large", model_version=version).inc()
            raise APIError(
                413,
                "batch_too_large",
                f"Batch exceeds {settings.max_batch_size} items.",
            )

        BATCH_SIZE.observe(len(payload.texts))
        results: list[BatchItem | None] = [None] * len(payload.texts)
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for index, text in enumerate(payload.texts):
            try:
                _validate_text(text, settings, version)
            except APIError as exc:
                results[index] = BatchItem(index=index, error=exc.message)
            else:
                valid_indices.append(index)
                valid_texts.append(text)

        if valid_texts:
            predictions, latency_ms = await _run_predictions(runtime, valid_texts)
            for index, text, prediction in zip(valid_indices, valid_texts, predictions):
                _instrument_prediction(runtime, text, prediction)
                results[index] = BatchItem(
                    index=index,
                    prediction=_prediction_response(runtime, prediction, latency_ms),
                )
        return BatchResponse(results=[item for item in results if item is not None])

    @router.get("/model/info", response_model=ModelInfo, responses={503: error_responses[503]})
    async def model_info(request: Request) -> ModelInfo:
        runtime: InferenceRuntime = request.app.state.runtime
        predictor = runtime.predictor
        if predictor is None:
            raise APIError(503, "model_not_ready", "No model is loaded.")
        return ModelInfo(
            model_version=predictor.version,
            stage=str(getattr(predictor, "stage", settings.model_stage)),
            predictor_class=type(predictor).__name__,
            metrics=dict(getattr(predictor, "metrics", {})),
            fairness_delta=getattr(predictor, "fairness_delta", None),
            trained_at=getattr(predictor, "trained_at", None),
            run_id=getattr(predictor, "run_id", None),
            build_revision=settings.build_revision,
        )

    @router.post(
        "/explain",
        response_model=ExplainResponse,
        responses={**error_responses, 501: {"model": ErrorResponse}},
    )
    async def explain(request: Request, payload: ExplainRequest) -> ExplainResponse:
        runtime: InferenceRuntime = request.app.state.runtime
        _validate_text(payload.text, settings, _model_version(runtime))
        active_explainer = request.app.state.explainer
        if active_explainer is None:
            raise APIError(
                501,
                "explainer_not_available",
                "The LIME explainer has not been installed yet.",
            )
        explanation = await asyncio.to_thread(
            active_explainer.explain, payload.text, payload.method
        )
        return ExplainResponse(
            method=explanation.method,
            label=explanation.label,
            score=explanation.score,
            model_version=_model_version(runtime),
            attributions=[
                Attribution(token=item.token, attribution=item.attribution)
                for item in explanation.attributions
            ],
        )

    app.include_router(router)
    return app
