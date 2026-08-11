"""Application operations endpoint tests."""

import pytest
from fastapi.testclient import TestClient

from sentiment.config import Settings, get_settings
from sentiment.models import registry
from sentiment.serving.app import create_app
from sentiment.serving.predictor import StubPredictor


def test_health_ready_metrics_and_openapi(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert health.headers["x-request-id"]

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model_version": "stub-0"}

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "http_requests_total" in metrics.text
    assert "ml_model_loaded 1.0" in metrics.text

    specification = client.get("/openapi.json").json()
    assert specification["info"]["title"] == "Sentiment Service"
    assert "/ready" in specification["paths"]


def test_unknown_route_and_validation_errors_are_typed(client: TestClient) -> None:
    missing = client.get("/api/v1/nope")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "not_found"
    assert missing.json()["request_id"] == missing.headers["x-request-id"]

    invalid = client.post("/api/v1/predict", json={})
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "validation_error"


def test_caller_request_id_is_preserved(client: TestClient) -> None:
    response = client.get("/health", headers={"x-request-id": "demo-request"})
    assert response.headers["x-request-id"] == "demo-request"


def test_failed_model_load_keeps_health_up_and_readiness_down() -> None:
    def fail_to_load(_settings: Settings) -> None:
        raise RuntimeError("registry unavailable")

    with TestClient(create_app(predictor_factory=fail_to_load)) as client:
        assert client.get("/health").status_code == 200
        response = client.get("/ready")
        assert response.status_code == 503
        assert response.json()["error_code"] == "model_not_ready"


def test_registry_backend_uses_m2_loader_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: dict[str, str] = {}

    def load_production_predictor(*, tracking_uri: str, stage: str) -> StubPredictor:
        received.update(tracking_uri=tracking_uri, stage=stage)
        return StubPredictor()

    monkeypatch.setattr(
        registry,
        "load_production_predictor",
        load_production_predictor,
        raising=False,
    )
    monkeypatch.setenv("SENTIMENT_PREDICTOR_BACKEND", "registry")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/ready").status_code == 200
            assert received == {
                "tracking_uri": "http://localhost:5000",
                "stage": "Production",
            }
    finally:
        get_settings.cache_clear()
