#!/usr/bin/env python3
"""Demo script to prove the monitoring stack is capturing API metrics.

This is intentionally simple and presentation-friendly. It:
1) checks the API is ready,
2) sends one real prediction request,
3) queries Prometheus for evidence that a scrape is active,
4) prints a clear pass/fail result.

Usage:
    python scripts/demo_prometheus_monitoring.py
    API_BASE=http://localhost:8000 PROMETHEUS_BASE=http://localhost:9090 \
        python scripts/demo_prometheus_monitoring.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
PROMETHEUS_BASE = os.environ.get("PROMETHEUS_BASE", "http://localhost:9090")


def fetch_json(url: str, *, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = httpx.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


def prometheus_query(query: str) -> list[dict[str, Any]]:
    payload = fetch_json(f"{PROMETHEUS_BASE}/api/v1/query", params={"query": query})
    result: list[dict[str, Any]] = payload.get("data", {}).get("result", [])
    return result


def main() -> int:
    print("== Monitoring demo ==")
    print(f"API: {API_BASE}")
    print(f"Prometheus: {PROMETHEUS_BASE}")
    print()

    try:
        ready = fetch_json(f"{API_BASE}/ready")
        print(f"[OK] API ready: {ready}")
    except Exception as exc:  # pragma: no cover - demo script
        print(f"[FAIL] API is not responding: {exc}")
        return 1

    try:
        prediction = httpx.post(
            f"{API_BASE}/api/v1/predict",
            json={"text": "Excellent build quality and fast service."},
            timeout=10,
        )
        prediction.raise_for_status()
        payload = prediction.json()
        print(
            f"[OK] Prediction sample: label={payload.get('label')}, "
            f"confidence={payload.get('confidence')}"
        )
    except Exception as exc:  # pragma: no cover - demo script
        print(f"[FAIL] Prediction request failed: {exc}")
        return 1

    checks = [
        ('up{job="sentiment-api"}', "Prometheus scrape status"),
        ("http_requests_total", "HTTP request counter"),
        ("ml_predictions_total", "ML prediction counter"),
    ]

    failures: list[str] = []
    for query, label in checks:
        result = prometheus_query(query)
        samples = len(result)
        print(f"[CHECK] {label}: {samples} sample(s)")
        if samples == 0:
            failures.append(label)

    if failures:
        print()
        print("[FAIL] Prometheus has no data for:", ", ".join(failures))
        print("Check that the API is running and the Prometheus scrape target is healthy.")
        return 1

    print()
    print("[PASS] Prometheus is scraping the API and the monitoring metrics are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
