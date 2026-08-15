"""Application operations endpoint tests."""

import logging

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
                "tracking_uri": get_settings().mlflow_tracking_uri,
                "stage": get_settings().model_stage,
            }
    finally:
        get_settings.cache_clear()


def test_token_protected_reload_atomically_updates_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = iter(["v1", "v2"])

    def load(_settings: Settings) -> StubPredictor:
        predictor = StubPredictor()
        predictor.version = next(versions)
        return predictor

    monkeypatch.setenv("SENTIMENT_RELOAD_TOKEN", "test-reload-secret")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(predictor_factory=load)) as reload_client:
            unauthorized = reload_client.post("/reload")
            assert unauthorized.status_code == 401
            response = reload_client.post(
                "/reload", headers={"x-reload-token": "test-reload-secret"}
            )
            assert response.status_code == 200
            assert response.json() == {"status": "reloaded", "model_version": "v2"}
            assert reload_client.get("/ready").json()["model_version"] == "v2"
    finally:
        get_settings.cache_clear()


def test_failed_reload_keeps_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def load(_settings: Settings) -> StubPredictor:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("replacement corrupt")
        predictor = StubPredictor()
        predictor.version = "stable"
        return predictor

    monkeypatch.setenv("SENTIMENT_RELOAD_TOKEN", "test-reload-secret")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(predictor_factory=load)) as reload_client:
            response = reload_client.post(
                "/reload", headers={"x-reload-token": "test-reload-secret"}
            )
            assert response.status_code == 503
            assert response.json()["error_code"] == "reload_failed"
            assert reload_client.get("/ready").json()["model_version"] == "stable"
    finally:
        get_settings.cache_clear()


def test_request_body_limit_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_MAX_REQUEST_BODY_BYTES", "20")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as limited_client:
            response = limited_client.post("/api/v1/predict", json={"text": "x" * 100})
            assert response.status_code == 413
            assert response.json()["error_code"] == "request_too_large"
            assert response.headers["x-request-id"] == response.json()["request_id"]
    finally:
        get_settings.cache_clear()


def test_inference_failure_is_logged_not_just_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 500 says nothing about why, so the cause has to survive in the log.

    A tokenizer that raised under concurrent load once produced a wall of
    ``prediction_failed`` responses and an empty log, which made the cause
    invisible until it was reproduced by hand.
    """

    class _BrokenPredictor(StubPredictor):
        """Warms up cleanly, then fails — the shape a concurrency bug takes."""

        version = "broken-1"

        def __init__(self) -> None:
            super().__init__()
            self.warmed = False

        def predict(self, texts: object) -> list:
            if not self.warmed:
                self.warmed = True
                return super().predict(list(texts))  # type: ignore[arg-type]
            raise RuntimeError("Already borrowed")

    def load(_settings: Settings) -> StubPredictor:
        return _BrokenPredictor()

    with TestClient(create_app(predictor_factory=load), raise_server_exceptions=False) as broken:
        with caplog.at_level(logging.ERROR, logger="sentiment.serving.app"):
            response = broken.post("/api/v1/predict", json={"text": "giáo viên dạy hay"})

    assert response.status_code == 500
    assert response.json()["error_code"] == "prediction_failed"
    assert "Already borrowed" in caplog.text
    assert "broken-1" in caplog.text
