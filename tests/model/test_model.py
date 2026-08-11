"""Unit tests for baseline model, transformer predictor structure, and evaluation metrics."""

import pytest
from src.sentiment.config import settings
from src.sentiment.serving.predictor import StubPredictor
from src.sentiment.models.baseline import BaselinePredictor
from src.sentiment.training.evaluate import check_latency_budget, evaluate_predictions


def test_stub_predictor():
    predictor = StubPredictor()
    result = predictor.predict("Giáo viên dạy rất hay và nhiệt tình.")

    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.5
    assert "NEGATIVE" in result["probabilities"]
    assert "NEUTRAL" in result["probabilities"]
    assert "POSITIVE" in result["probabilities"]


def test_baseline_predictor():
    texts = [
        "thầy dạy rất hay và nhiệt tình .",
        "môn học dở tệ , bài giảng nhàm chán .",
        "phòng học bình thường .",
    ]
    labels = [2, 0, 1]  # positive, negative, neutral

    model = BaselinePredictor()
    model.fit(texts, labels)

    res = model.predict("thầy dạy rất hay .")
    assert res["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3


def test_evaluate_predictions():
    y_true = [2, 0, 1, 2, 0]
    y_pred = [2, 0, 1, 2, 1]

    metrics = evaluate_predictions(y_true, y_pred)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert metrics["accuracy"] == 0.8


def test_latency_budget():
    predictor = StubPredictor()
    latency_info = check_latency_budget(predictor, p95_target_ms=50.0, num_runs=10)

    assert "p95_latency_ms" in latency_info
    assert latency_info["passed_sla"] is True
