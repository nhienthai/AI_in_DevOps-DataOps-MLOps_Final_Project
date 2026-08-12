"""Unit tests for baseline models and evaluation metrics."""

import pytest

pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.serving.predictor import StubPredictor  # noqa: E402
from sentiment.training.evaluate import (  # noqa: E402
    check_latency_budget,
    evaluate_predictions,
)


def test_stub_predictor() -> None:
    predictor = StubPredictor()
    result = predictor.predict(["Giáo viên dạy rất hay và nhiệt tình."])[0]

    assert result.label in {"positive", "negative"}
    assert 0.0 <= result.score <= 1.0
    assert 0.5 <= result.confidence <= 1.0


def test_baseline_predictor() -> None:
    texts = [
        "thầy dạy rất hay và nhiệt tình.",
        "môn học dở tệ, bài giảng nhàm chán.",
        "phòng học bình thường.",
    ]
    labels = [2, 0, 1]

    model = BaselinePredictor()
    model.fit(texts, labels)

    result = model.predict(["thầy dạy rất hay."])[0]
    assert result.label in {"positive", "negative", "neutral"}
    assert 0.0 <= result.confidence <= 1.0


def test_evaluate_predictions() -> None:
    metrics = evaluate_predictions([2, 0, 1, 2, 0], [2, 0, 1, 2, 1])

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 0.8


def test_latency_budget() -> None:
    latency_info = check_latency_budget(
        StubPredictor(), p95_target_ms=50.0, num_runs=10
    )

    assert "p95_latency_ms" in latency_info
    assert latency_info["passed_sla"] is True
