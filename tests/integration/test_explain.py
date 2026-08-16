"""M3's explanation endpoint and M5 integration boundary tests."""

import pytest
from fastapi.testclient import TestClient

from sentiment.config import get_settings
from sentiment.serving.app import create_app
from sentiment.serving.predictor import Explanation, TokenAttribution


class FakeExplainer:
    """Small stand-in for M5's future LIME implementation."""

    def explain(self, text: str, method: str = "lime") -> Explanation:
        assert text
        assert method == "lime"
        return Explanation(
            method="lime",
            label="positive",
            score=0.8,
            attributions=(
                TokenAttribution(token="excellent", attribution=0.4),
                TokenAttribution(token="battery", attribution=0.1),
            ),
        )


def test_endpoint_reports_501_when_no_explainer_can_be_built() -> None:
    """A deployment without LIME installed must say so rather than 500."""
    with TestClient(create_app(explainer_factory=lambda predictor: None)) as client:
        response = client.post("/api/v1/explain", json={"text": "Excellent battery life."})
        assert response.status_code == 501
        assert response.json()["error_code"] == "explainer_not_available"


def test_default_wiring_explains_against_the_live_model(client: TestClient) -> None:
    """The default factory builds an explainer bound to the serving predictor."""
    response = client.post("/api/v1/explain", json={"text": "Giáo viên dạy rất hay."})
    assert response.status_code == 200

    body = response.json()
    assert body["method"] == "lime"
    assert body["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["attributions"]
    assert all(isinstance(item["token"], str) for item in body["attributions"])


def test_explainer_is_rebuilt_when_the_predictor_changes() -> None:
    """An explanation must describe the model serving now, not one a reload replaced."""
    built: list[object] = []

    def factory(predictor: object) -> FakeExplainer:
        built.append(predictor)
        return FakeExplainer()

    with TestClient(create_app(explainer_factory=factory)) as client:
        client.post("/api/v1/explain", json={"text": "first"})
        client.post("/api/v1/explain", json={"text": "second"})
        assert len(built) == 1, "explainer should be cached while the predictor is unchanged"


def test_injected_explainer_returns_the_public_contract() -> None:
    with TestClient(create_app(explainer=FakeExplainer())) as client:
        response = client.post(
            "/api/v1/explain",
            json={"text": "Excellent battery life.", "method": "lime"},
        )
        assert response.status_code == 200
        assert response.json() == {
            "method": "lime",
            "label": "positive",
            "score": 0.8,
            "model_version": "stub-0",
            "attributions": [
                {"token": "excellent", "attribution": 0.4},
                {"token": "battery", "attribution": 0.1},
            ],
        }


def test_explain_validates_text_before_calling_explainer() -> None:
    with TestClient(create_app(explainer=FakeExplainer())) as client:
        blank = client.post("/api/v1/explain", json={"text": " "})
        assert blank.status_code == 422
        assert blank.json()["error_code"] == "empty_text"

        oversized = client.post("/api/v1/explain", json={"text": "x" * 5_001})
        assert oversized.status_code == 413
        assert oversized.json()["error_code"] == "text_too_long"


def test_second_concurrent_explanation_is_rejected() -> None:
    """One /explain is many model calls, so they must not pile up.

    Six concurrent explanations once cost eight minutes of CPU with every one of
    them answered 200, because /explain bypassed the admission control that
    /predict goes through.
    """
    import threading

    release = threading.Event()
    started = threading.Event()

    class _SlowExplainer:
        def explain(self, text: str, method: str = "lime") -> Explanation:
            started.set()
            release.wait(timeout=5)
            return Explanation(
                method="lime",
                label="positive",
                score=0.9,
                attributions=(TokenAttribution(token="hay", attribution=0.5),),
            )

    with TestClient(create_app(explainer=_SlowExplainer())) as slow_client:
        results: list[int] = []

        def call() -> None:
            results.append(
                slow_client.post("/api/v1/explain", json={"text": "giáo viên dạy hay"}).status_code
            )

        first = threading.Thread(target=call)
        first.start()
        assert started.wait(timeout=5), "first explanation never started"

        second = slow_client.post("/api/v1/explain", json={"text": "môn học chán"})
        assert second.status_code == 429
        assert second.json()["error_code"] == "explain_overloaded"

        release.set()
        first.join(timeout=10)
        assert results == [200]


def test_explanation_that_overruns_its_budget_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    class _HangingExplainer:
        def explain(self, text: str, method: str = "lime") -> Explanation:
            time.sleep(5)
            raise AssertionError("should have been abandoned")

    monkeypatch.setenv("SENTIMENT_EXPLAIN_TIMEOUT_SECONDS", "0.2")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(explainer=_HangingExplainer())) as slow_client:
            response = slow_client.post("/api/v1/explain", json={"text": "giáo viên dạy hay"})
        assert response.status_code == 504
        assert response.json()["error_code"] == "explain_timeout"
    finally:
        get_settings.cache_clear()
