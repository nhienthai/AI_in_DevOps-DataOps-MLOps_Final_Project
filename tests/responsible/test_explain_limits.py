"""Bounds on the cost of one explanation.

LIME turns a single request into ``num_samples`` model calls. Handing all of them
to the predictor in one call bypassed every limit the serving layer has — batch
size, the inference semaphore, and the timeout — which with a transformer meant
one unauthenticated request costing 80 seconds of CPU and six of them costing
eight minutes, with nothing rejected.
"""

from typing import Any, Sequence

import pytest

pytest.importorskip("lime")

from sentiment.responsible.explain import LimeExplainer  # noqa: E402
from sentiment.serving.predictor import Prediction  # noqa: E402


class _CountingPredictor:
    """Records the size of every batch the explainer asks for."""

    version = "counting-1"

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        self.batch_sizes.append(len(texts))
        return [
            Prediction(label="positive", score=0.9, confidence=0.9, truncated=False) for _ in texts
        ]


def test_perturbations_are_chunked_to_the_batch_limit() -> None:
    """No single model call may exceed what /predict/batch would allow."""
    predictor: Any = _CountingPredictor()
    explainer = LimeExplainer(predictor, num_features=4, num_samples=200, batch_size=64)

    explainer.explain("giáo viên dạy rất hay")

    assert predictor.batch_sizes, "explainer never called the model"
    assert max(predictor.batch_sizes) <= 64, predictor.batch_sizes


def test_total_work_is_bounded_by_num_samples() -> None:
    """Chunking must not silently multiply the number of model evaluations."""
    predictor: Any = _CountingPredictor()
    explainer = LimeExplainer(predictor, num_features=4, num_samples=120, batch_size=32)

    explainer.explain("giáo viên dạy rất hay")

    # One call scores the input itself; the rest are LIME's perturbations.
    assert sum(predictor.batch_sizes) <= 120 + 1


def test_batch_size_defaults_to_one_call() -> None:
    """Without a limit the explainer keeps its original single-call behaviour."""
    predictor: Any = _CountingPredictor()
    explainer = LimeExplainer(predictor, num_features=4, num_samples=80)

    explainer.explain("giáo viên dạy rất hay")

    assert max(predictor.batch_sizes) > 1
