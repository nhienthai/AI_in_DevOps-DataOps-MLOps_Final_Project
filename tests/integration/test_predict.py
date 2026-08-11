"""Inference endpoint contract and instrumentation tests."""

import pytest
from fastapi.testclient import TestClient

from sentiment.config import Settings
from sentiment.serving.app import create_app
from sentiment.serving.predictor import Prediction


def test_predict_returns_complete_consistent_contract(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"text": "Great product"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "label",
        "score",
        "confidence",
        "model_version",
        "truncated",
        "latency_ms",
    }
    assert body["confidence"] == pytest.approx(max(body["score"], 1 - body["score"]))
    assert body["model_version"] == "stub-0"


def test_blank_and_oversized_text_have_typed_errors(client: TestClient) -> None:
    blank = client.post("/api/v1/predict", json={"text": "   "})
    assert blank.status_code == 422
    assert blank.json()["error_code"] == "empty_text"

    oversized = client.post("/api/v1/predict", json={"text": "x" * 5_001})
    assert oversized.status_code == 413
    assert oversized.json()["error_code"] == "text_too_long"


def test_extra_request_fields_are_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"text": "valid", "unexpected": True})
    assert response.status_code == 422


def test_unicode_and_emoji_are_preserved_by_the_contract(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"text": "ดีมาก 😊"})
    assert response.status_code == 200
    assert response.json()["label"] in {"positive", "negative"}


def test_batch_returns_ordered_results_and_isolates_bad_items(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/predict/batch",
        json={"texts": ["good", "   ", "x" * 5_001, "bad"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [item["index"] for item in results] == [0, 1, 2, 3]
    assert results[0]["prediction"] is not None
    assert results[1]["error"] == "Text must not be blank."
    assert "5000" in results[2]["error"]
    assert results[3]["prediction"] is not None


def test_batch_limits_are_enforced(client: TestClient) -> None:
    empty = client.post("/api/v1/predict/batch", json={"texts": []})
    assert empty.status_code == 422

    oversized = client.post("/api/v1/predict/batch", json={"texts": ["x"] * 65})
    assert oversized.status_code == 413
    assert oversized.json()["error_code"] == "batch_too_large"


def test_model_info_reports_stub_provenance(client: TestClient) -> None:
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    assert response.json() == {
        "model_version": "stub-0",
        "stage": "Development",
        "predictor_class": "StubPredictor",
        "metrics": {},
        "fairness_delta": None,
        "trained_at": None,
        "run_id": None,
    }


def test_prediction_and_http_metrics_use_bounded_labels(client: TestClient) -> None:
    client.post("/api/v1/predict", json={"text": "measured"})
    client.post("/api/v1/predict/batch", json={"texts": ["a", " "]})
    metrics = client.get("/metrics").text
    assert 'model_version="stub-0"' in metrics
    assert "ml_input_length_chars_count" in metrics
    assert "ml_prediction_confidence_count" in metrics
    assert "ml_batch_prediction_size_count" in metrics
    assert "ml_prediction_errors_total" in metrics
    assert 'endpoint="/api/v1/predict"' in metrics
    assert 'endpoint="unmatched"' in metrics


def test_inference_failure_is_typed_and_counted() -> None:
    class BrokenPredictor:
        version = "broken-1"

        def predict(self, _texts: list[str]) -> list[Prediction]:
            raise RuntimeError("secret implementation detail")

    def factory(_settings: Settings) -> BrokenPredictor:
        return BrokenPredictor()

    with TestClient(create_app(predictor_factory=factory)) as client:
        response = client.post("/api/v1/predict", json={"text": "hello"})
        assert response.status_code == 500
        assert response.json()["error_code"] == "prediction_failed"
        assert "secret" not in response.text
        assert 'model_version="broken-1"' in client.get("/metrics").text


def test_all_inference_routes_appear_in_openapi(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/api/v1/predict",
        "/api/v1/predict/batch",
        "/api/v1/model/info",
        "/api/v1/explain",
    } <= set(paths)
