"""M3's explanation endpoint and M5 integration boundary tests."""

from fastapi.testclient import TestClient

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


def test_endpoint_is_explicitly_unavailable_until_m5_plugs_in(
    client: TestClient,
) -> None:
    response = client.post("/api/v1/explain", json={"text": "Excellent battery life."})
    assert response.status_code == 501
    assert response.json()["error_code"] == "explainer_not_available"


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
