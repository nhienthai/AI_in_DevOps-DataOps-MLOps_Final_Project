"""End-to-end checks against a running stack.

Run with: docker compose up -d && pytest tests/integration -m integration
"""

import time

import httpx
import pytest

pytestmark = pytest.mark.integration

API = "http://localhost:8000"
PROM = "http://localhost:9090"


def test_api_is_ready():
    body = httpx.get(f"{API}/ready", timeout=10).json()
    assert body["status"] == "ready"


def test_api_serves_a_prediction():
    body = httpx.post(
        f"{API}/api/v1/predict", json={"text": "Excellent build quality."}, timeout=10
    ).json()
    assert body["label"] in {"positive", "neutral", "negative"}
    assert body["confidence"] >= 0.5


def test_prometheus_has_scraped_the_api():
    deadline = time.monotonic() + 30
    latest_result = None
    while time.monotonic() < deadline:
        latest_result = httpx.get(
            f"{PROM}/api/v1/query",
            params={"query": 'up{job="sentiment-api"}'},
            timeout=10,
        ).json()
        samples = latest_result.get("data", {}).get("result", [])
        if samples and samples[0]["value"][1] == "1":
            return
        time.sleep(1)

    pytest.fail(f"Prometheus did not scrape the API before timeout: {latest_result}")


def test_alert_rules_are_loaded():
    rules = httpx.get(f"{PROM}/api/v1/rules", timeout=10).json()
    names = {rule["name"] for group in rules["data"]["groups"] for rule in group["rules"]}
    assert {"APIDown", "HighErrorRate", "DriftDetected"} <= names
