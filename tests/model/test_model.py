"""Unit tests for model contracts and evaluation metrics."""

import json
from pathlib import Path

import pytest

pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.models.transformer import _validate_model_label_map  # noqa: E402
from sentiment.serving.predictor import StubPredictor  # noqa: E402
from sentiment.training.evaluate import check_latency_budget, evaluate_predictions  # noqa: E402


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
    latency_info = check_latency_budget(StubPredictor(), p95_target_ms=50.0, num_runs=10)

    assert "p95_latency_ms" in latency_info
    assert latency_info["passed_sla"] is True


def test_transformer_label_metadata_must_match_serving_contract() -> None:
    expected = {0: "negative", 1: "neutral", 2: "positive"}
    _validate_model_label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, expected)
    _validate_model_label_map({0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}, expected)

    with pytest.raises(RuntimeError, match="does not match serving contract"):
        _validate_model_label_map({0: "positive", 1: "neutral", 2: "negative"}, expected)


def test_kaggle_notebook_is_clean_and_reproducible() -> None:
    notebook_path = Path(__file__).parents[2] / "notebooks" / "train-xlm-roberta.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["metadata"]["kaggle"]["isInternetEnabled"] is True
    assert notebook["metadata"]["kaggle"]["isGpuEnabled"] is True
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "--branch', 'main'" in sources
    assert "'pip', 'install', '-q', '-e', '.'" in sources
