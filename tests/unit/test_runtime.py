"""Inference runtime lifecycle, overload, and atomic reload tests."""

import asyncio
import time

import pytest

from sentiment.config import Settings
from sentiment.serving.metrics import DriftReference
from sentiment.serving.predictor import Prediction
from sentiment.serving.runtime import (
    InferenceOverloadedError,
    InferenceRuntime,
    InferenceTimeoutError,
)

REFERENCE = DriftReference((0.0, 10.0, 100.0), (0.5, 0.5), 0.5)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class ControlledPredictor:
    stage = "Production"

    def __init__(self, version: str, delay: float = 0.0) -> None:
        self.version = version
        self.delay = delay
        self.calls = 0

    def predict(self, texts: list[str]) -> list[Prediction]:
        self.calls += 1
        if self.calls > 1 and self.delay:
            time.sleep(self.delay)
        return [Prediction("neutral", 0.25, 0.6, False) for _ in texts]


@pytest.mark.anyio
async def test_reload_is_atomic_and_retains_old_model_on_failure() -> None:
    loaded = [ControlledPredictor("v1"), ControlledPredictor("v2")]

    def loader(_settings: Settings) -> ControlledPredictor:
        if loaded:
            return loaded.pop(0)
        raise RuntimeError("registry unavailable")

    runtime = InferenceRuntime(Settings(), loader, REFERENCE)
    assert await runtime.start()
    assert runtime.predictor is not None and runtime.predictor.version == "v1"
    assert await runtime.reload()
    assert runtime.predictor is not None and runtime.predictor.version == "v2"
    assert not await runtime.reload()
    assert runtime.predictor is not None and runtime.predictor.version == "v2"
    await runtime.close()


@pytest.mark.anyio
async def test_queue_budget_rejects_excess_concurrency() -> None:
    predictor = ControlledPredictor("slow", delay=0.15)
    settings = Settings(max_concurrent_inferences=1, queue_timeout_seconds=0.01)
    runtime = InferenceRuntime(settings, lambda _settings: predictor, REFERENCE)
    assert await runtime.start()

    first = asyncio.create_task(runtime.predict(["first"]))
    await asyncio.sleep(0.01)
    with pytest.raises(InferenceOverloadedError):
        await runtime.predict(["second"])
    await first
    await runtime.close()


@pytest.mark.anyio
async def test_inference_timeout_releases_capacity_after_worker_finishes() -> None:
    predictor = ControlledPredictor("slow", delay=0.05)
    settings = Settings(
        max_concurrent_inferences=1,
        inference_timeout_seconds=0.01,
        queue_timeout_seconds=0.01,
    )
    runtime = InferenceRuntime(settings, lambda _settings: predictor, REFERENCE)
    assert await runtime.start()
    with pytest.raises(InferenceTimeoutError):
        await runtime.predict(["slow request"])
    await asyncio.sleep(0.1)
    predictor.delay = 0
    runtime.settings.inference_timeout_seconds = 0.2
    predictions, _elapsed, _queued = await runtime.predict(["recovered"])
    assert predictions[0].label == "neutral"
    await runtime.close()
