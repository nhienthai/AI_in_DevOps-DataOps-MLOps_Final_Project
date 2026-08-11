"""FastAPI application factory for the sentiment serving service."""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import APIRouter, FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentiment.config import Settings, get_settings
from sentiment.serving.errors import APIError, install_error_handlers
from sentiment.serving.metrics import (
    BATCH_SIZE,
    DRIFT_PSI,
    INPUT_LENGTH,
    LOW_CONFIDENCE,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    MODEL_LOADED,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
    DriftReference,
    DriftTracker,
)
from sentiment.serving.middleware import MetricsMiddleware
from sentiment.serving.predictor import Explainer, Prediction, Predictor, StubPredictor
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
)

PredictorFactory = Callable[[Settings], Predictor]

_BOOTSTRAP_REFERENCE = DriftReference(
    length_bin_edges=(0.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 100_000.0),
    length_bin_freqs=(0.1, 0.2, 0.3, 0.2, 0.15, 0.05),
    positive_prior=0.5,
)


def _default_predictor_factory(settings: Settings) -> Predictor:
    """Load the selected backend while keeping M2 behind a narrow boundary."""
    if settings.predictor_backend == "stub":
        return StubPredictor(max_chars=settings.max_text_length)

    from sentiment.models import registry

    loader = getattr(registry, "load_production_predictor", None)
    if loader is None:
        raise RuntimeError(
            "registry backend requires sentiment.models.registry.load_production_predictor"
        )
    predictor = loader(
        tracking_uri=settings.mlflow_tracking_uri,
        stage=settings.model_stage,
    )
    if not isinstance(predictor, Predictor):
        raise TypeError("registry loader did not return a Predictor")
    return predictor


def _model_version(app: FastAPI) -> str:
    """Return a safe metric label even when the model is unavailable."""
    predictor = getattr(app.state, "predictor", None)
    return str(getattr(predictor, "version", "unloaded"))


def create_app(
    predictor_factory: PredictorFactory | None = None,
    explainer: Explainer | None = None,
) -> FastAPI:
    """Build an isolated application instance for Uvicorn or tests."""
    settings = get_settings()
    load_predictor = predictor_factory or _default_predictor_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load serving dependencies before accepting inference traffic."""
        app.state.predictor = None
        app.state.explainer = explainer
        app.state.model_load_error = None
        MODEL_LOADED.set(0)

        try:
            predictor = load_predictor(settings)
            reference = getattr(predictor, "drift_reference", _BOOTSTRAP_REFERENCE)
            app.state.predictor = predictor
            app.state.drift = DriftTracker(
                cast(DriftReference, reference),
                window_size=settings.drift_window_size,
            )
        except Exception as exc:  # readiness reports failure without killing liveness
            app.state.model_load_error = str(exc)
        else:
            stage = str(getattr(predictor, "stage", settings.model_stage))
            MODEL_LOADED.set(1)
            MODEL_INFO.info(
                {
                    "version": predictor.version,
                    "predictor_class": type(predictor).__name__,
                    "stage": stage,
                }
            )
            MODEL_LAST_RELOAD.set(time.time())

        yield
        app.state.predictor = None
        MODEL_LOADED.set(0)

    app = FastAPI(
        title="Sentiment Service",
        version="0.1.0",
        description="Real-time sentiment analysis for e-commerce reviews.",
        lifespan=lifespan,
    )
    app.add_middleware(MetricsMiddleware)
    install_error_handlers(app)

    error_responses: dict[int | str, dict[str, Any]] = {
        413: {"model": ErrorResponse, "description": "Configured size limit exceeded."},
        422: {"model": ErrorResponse, "description": "Request validation failed."},
        503: {"model": ErrorResponse, "description": "Model is not ready."},
    }

    @app.get("/health", tags=["operations"])
    async def health() -> dict[str, str]:
        """Report process liveness independently from model readiness."""
        return {"status": "ok"}

    @app.get("/ready", tags=["operations"], responses={503: error_responses[503]})
    async def ready(request: Request) -> dict[str, str]:
        """Report readiness only after a model is loaded."""
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise APIError(503, "model_not_ready", "No model is loaded.")
        return {"status": "ready", "model_version": predictor.version}

    @app.get("/metrics", tags=["operations"], include_in_schema=True)
    async def metrics() -> Response:
        """Expose Prometheus metrics in the standard text format."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    router = APIRouter(prefix="/api/v1", tags=["inference"])

    def score_one(request: Request, text: str) -> tuple[Prediction, float]:
        """Validate, predict, and instrument one item."""
        predictor = getattr(request.app.state, "predictor", None)
        version = _model_version(request.app)
        if predictor is None:
            PREDICTION_ERRORS.labels(error_type="model_not_ready", model_version=version).inc()
            raise APIError(503, "model_not_ready", "No model is loaded.")
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

        started = time.perf_counter()
        try:
            predictions = predictor.predict([text])
            if len(predictions) != 1:
                raise RuntimeError("predictor returned an unexpected result count")
            prediction = predictions[0]
        except Exception as exc:
            PREDICTION_ERRORS.labels(error_type="inference_error", model_version=version).inc()
            raise APIError(500, "prediction_failed", "Prediction failed.") from exc
        elapsed = time.perf_counter() - started

        PREDICTION_LATENCY.labels(model_version=version).observe(elapsed)
        PREDICTION_COUNT.labels(label=prediction.label, model_version=version).inc()
        PREDICTION_CONFIDENCE.observe(prediction.confidence)
        INPUT_LENGTH.observe(len(text))
        if prediction.confidence < settings.low_confidence_threshold:
            LOW_CONFIDENCE.inc()
        request.app.state.drift.observe(len(text))
        DRIFT_PSI.set(request.app.state.drift.psi())
        return prediction, elapsed * 1_000.0

    def prediction_response(
        request: Request, prediction: Prediction, latency_ms: float
    ) -> PredictResponse:
        """Convert an internal prediction to its public representation."""
        return PredictResponse(
            label=prediction.label,
            score=prediction.score,
            confidence=prediction.confidence,
            model_version=_model_version(request.app),
            truncated=prediction.truncated,
            latency_ms=round(latency_ms, 3),
        )

    @router.post(
        "/predict",
        response_model=PredictResponse,
        responses=error_responses,
        summary="Classify one review",
    )
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        """Classify a single review as positive or negative."""
        prediction, latency_ms = score_one(request, payload.text)
        return prediction_response(request, prediction, latency_ms)

    @router.post(
        "/predict/batch",
        response_model=BatchResponse,
        responses=error_responses,
        summary="Classify up to 64 reviews",
    )
    async def predict_batch(request: Request, payload: BatchRequest) -> BatchResponse:
        """Classify a batch while isolating invalid individual items."""
        if len(payload.texts) > settings.max_batch_size:
            PREDICTION_ERRORS.labels(
                error_type="batch_too_large",
                model_version=_model_version(request.app),
            ).inc()
            raise APIError(
                413,
                "batch_too_large",
                f"Batch exceeds {settings.max_batch_size} items.",
            )

        BATCH_SIZE.observe(len(payload.texts))
        results: list[BatchItem] = []
        for index, text in enumerate(payload.texts):
            try:
                item_prediction, latency_ms = score_one(request, text)
            except APIError as exc:
                results.append(BatchItem(index=index, error=exc.message))
            else:
                results.append(
                    BatchItem(
                        index=index,
                        prediction=prediction_response(request, item_prediction, latency_ms),
                    )
                )
        return BatchResponse(results=results)

    @router.get("/model/info", response_model=ModelInfo, responses={503: error_responses[503]})
    async def model_info(request: Request) -> ModelInfo:
        """Return model provenance and evaluation metadata."""
        predictor = getattr(request.app.state, "predictor", None)
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
        )

    @router.post(
        "/explain",
        response_model=ExplainResponse,
        responses={**error_responses, 501: {"model": ErrorResponse}},
        summary="Explain one model decision with LIME",
    )
    async def explain(request: Request, payload: ExplainRequest) -> ExplainResponse:
        """Delegate token attribution to M5's deployed-model explainer."""
        if not payload.text.strip():
            raise APIError(422, "empty_text", "Text must not be blank.")
        if len(payload.text) > settings.max_text_length:
            raise APIError(
                413,
                "text_too_long",
                f"Text exceeds {settings.max_text_length} characters.",
            )
        active_explainer = getattr(request.app.state, "explainer", None)
        if active_explainer is None:
            raise APIError(
                501,
                "explainer_not_available",
                "The LIME explainer has not been installed yet.",
            )
        explanation = active_explainer.explain(payload.text, payload.method)
        return ExplainResponse(
            method=explanation.method,
            label=explanation.label,
            score=explanation.score,
            model_version=_model_version(request.app),
            attributions=[
                Attribution(token=item.token, attribution=item.attribution)
                for item in explanation.attributions
            ],
        )

    app.include_router(router)
    return app
