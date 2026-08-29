"""The browser test console served at /ui.

The page exists to exercise the API by hand and to classify a CSV without writing
code. It is deliberately a page and not an upload endpoint: the file is parsed in
the browser, so nothing but text ever crosses the wire and the API contract stays
the eight endpoints it already documents.
"""

import pytest
from fastapi.testclient import TestClient

from sentiment.config import get_settings
from sentiment.serving.app import create_app


def test_ui_is_served_as_html(client: TestClient) -> None:
    response = client.get("/ui")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>" in response.text


def test_ui_talks_to_the_documented_endpoints(client: TestClient) -> None:
    """The page must drive the real API, not a private one built for it."""
    page = client.get("/ui").text

    assert "/api/v1/predict/batch" in page
    assert "/api/v1/model/info" in page


def test_ui_respects_the_batch_limit(client: TestClient) -> None:
    """A CSV of any size has to be chunked to what /predict/batch accepts."""
    page = client.get("/ui").text
    settings = get_settings()

    assert str(settings.max_batch_size) in page


def test_ui_stays_out_of_the_public_api_schema(client: TestClient) -> None:
    """A console for humans should not appear as an operation for machines."""
    schema = client.get("/openapi.json").json()

    assert "/ui" not in schema["paths"]


def test_ui_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment that does not want a console on the API port can drop it."""
    monkeypatch.setenv("SENTIMENT_ENABLE_UI", "false")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as disabled:
            assert disabled.get("/ui").status_code == 404
    finally:
        get_settings.cache_clear()
