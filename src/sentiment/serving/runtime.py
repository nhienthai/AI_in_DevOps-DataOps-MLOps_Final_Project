"""Model lifecycle and bounded non-blocking inference execution."""

import asyncio
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

from sentiment.config import Settings
from sentiment.serving.metrics import (
    INFERENCE_IN_PROGRESS,
    INFERENCE_OVERLOADS,
    INFERENCE_QUEUE_DEPTH,
    INFERENCE_QUEUE_LATENCY,
    INFERENCE_TIMEOUTS,
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    MODEL_LOAD_DURATION,
    MODEL_LOAD_FAILURES,
    MODEL_LOADED,
    DriftReference,
    DriftTracker,
)
from sentiment.serving.predictor import Prediction, Predictor, validate_predictions

PredictorFactory = Callable[[Settings], Predictor]


class ModelNotReadyError(RuntimeError):
    """No warmed model is available."""


class InferenceOverloadedError(RuntimeError):
    """No inference slot became available within the queue budget."""


class InferenceTimeoutError(RuntimeError):
    """Model execution exceeded its time budget."""


class InferenceRuntime:
    """Own the active model, atomic reloads, and a bounded inference pool."""

    def __init__(
        self,
        settings: Settings,
        loader: PredictorFactory,
        fallback_reference: DriftReference,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.fallback_reference = fallback_reference
        self.predictor: Predictor | None = None
        self.drift: DriftTracker | None = None
        self.model_load_error: str | None = None
        self._reload_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(settings.max_concurrent_inferences)
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_inferences,
            thread_name_prefix="sentiment-inference",
        )

    async def start(self) -> bool:
        """Load the first model; failure leaves liveness up and readiness down."""
        return await self.reload()

    def _load_and_warm(self) -> tuple[Predictor, DriftTracker]:
        started = time.perf_counter()
        try:
            predictor = self.loader(self.settings)
            if not isinstance(predictor, Predictor):
                raise TypeError("model loader did not return a Predictor")
            warmup = predictor.predict([self.settings.warmup_text])
            validate_predictions(warmup, 1)
            raw_reference = getattr(predictor, "drift_reference", self.fallback_reference)
            reference = DriftReference(
                length_bin_edges=tuple(raw_reference.length_bin_edges),
                length_bin_freqs=tuple(raw_reference.length_bin_freqs),
                positive_prior=float(raw_reference.positive_prior),
            )
            drift = DriftTracker(reference, window_size=self.settings.drift_window_size)
            return predictor, drift
        finally:
            MODEL_LOAD_DURATION.observe(time.perf_counter() - started)

    async def reload(self) -> bool:
        """Warm a replacement and atomically publish it, retaining the old model on failure."""
        async with self._reload_lock:
            loop = asyncio.get_running_loop()
            try:
                predictor, drift = await loop.run_in_executor(self._executor, self._load_and_warm)
            except Exception as exc:
                self.model_load_error = str(exc)
                MODEL_LOAD_FAILURES.inc()
                if self.predictor is None:
                    MODEL_LOADED.set(0)
                return False

            self.predictor = predictor
            self.drift = drift
            self.model_load_error = None
            stage = str(getattr(predictor, "stage", self.settings.model_stage))
            MODEL_LOADED.set(1)
            MODEL_INFO.info(
                {
                    "version": predictor.version,
                    "predictor_class": type(predictor).__name__,
                    "stage": stage,
                    "build_revision": self.settings.build_revision,
                }
            )
            MODEL_LAST_RELOAD.set(time.time())
            return True

    def _release_after_timeout(self, future: asyncio.Future[list[Prediction]]) -> None:
        INFERENCE_IN_PROGRESS.dec()
        self._slots.release()
        try:
            future.result()
        except (Exception, asyncio.CancelledError):
            pass

    async def predict(self, texts: Sequence[str]) -> tuple[list[Prediction], float, float]:
        """Run one model batch without blocking the event loop."""
        predictor = self.predictor
        if predictor is None:
            raise ModelNotReadyError("No model is loaded.")

        queued = time.perf_counter()
        INFERENCE_QUEUE_DEPTH.inc()
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self.settings.queue_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            INFERENCE_OVERLOADS.inc()
            raise InferenceOverloadedError("Inference capacity is full.") from exc
        finally:
            INFERENCE_QUEUE_DEPTH.dec()
        queue_seconds = time.perf_counter() - queued
        INFERENCE_QUEUE_LATENCY.observe(queue_seconds)

        INFERENCE_IN_PROGRESS.inc()
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, predictor.predict, list(texts))
        try:
            predictions = await asyncio.wait_for(
                asyncio.shield(future), timeout=self.settings.inference_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            INFERENCE_TIMEOUTS.inc()
            future.add_done_callback(self._release_after_timeout)
            raise InferenceTimeoutError("Prediction exceeded its time limit.") from exc
        except Exception:
            INFERENCE_IN_PROGRESS.dec()
            self._slots.release()
            raise
        else:
            INFERENCE_IN_PROGRESS.dec()
            self._slots.release()

        validate_predictions(predictions, len(texts))
        return predictions, time.perf_counter() - started, queue_seconds

    async def close(self) -> None:
        """Stop accepting work and release executor resources."""
        self.predictor = None
        self.drift = None
        MODEL_LOADED.set(0)
        self._executor.shutdown(wait=False, cancel_futures=True)
