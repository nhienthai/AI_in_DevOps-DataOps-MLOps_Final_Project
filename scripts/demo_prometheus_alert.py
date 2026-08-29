#!/usr/bin/env python3
"""Force a real Prometheus alert for a demo.

This script temporarily injects one dead target into the same `sentiment-api`
job so Prometheus emits `up{job="sentiment-api"} == 0` and the `APIDown` rule
fires after the configured `for` period.

Usage:
    python scripts/demo_prometheus_alert.py
    python scripts/demo_prometheus_alert.py --timeout 150
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

PROMETHEUS_BASE = "http://localhost:9090"
PROM_CONFIG = Path("prometheus/prometheus.yml")


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def fetch_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    response = httpx.get(url, params=params, timeout=10)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    return payload


def fetch_alerts() -> list[dict[str, Any]]:
    data = fetch_json(f"{PROMETHEUS_BASE}/api/v1/alerts")
    alerts: list[dict[str, Any]] = data.get("data", {}).get("alerts", [])
    return alerts


def wait_for_ready() -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            response = httpx.get("http://localhost:8000/ready", timeout=3)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("API never returned to ready state after the demo")


def backup_and_inject_dead_target() -> None:
    original = PROM_CONFIG.read_text()
    backup = PROM_CONFIG.with_suffix(".yml.bak")
    backup.write_text(original)

    scrape_block = (
        "  - job_name: sentiment-api\n" "    metrics_path: /metrics\n" "    static_configs:\n"
    )
    live_only = f'{scrape_block}      - targets: ["api:8000"]\n'
    with_dead = f'{scrape_block}      - targets: ["api:8000", "127.0.0.1:65535"]\n'
    replacement = original.replace(live_only, with_dead)
    if replacement == original:
        raise RuntimeError("Failed to inject a dead target; check the Prometheus config format.")
    PROM_CONFIG.write_text(replacement)
    run(["docker", "compose", "exec", "prometheus", "kill", "-HUP", "1"])


def restore_original_config() -> None:
    backup = PROM_CONFIG.with_suffix(".yml.bak")
    if backup.exists():
        shutil.copy2(backup, PROM_CONFIG)
        backup.unlink()
        run(["docker", "compose", "exec", "prometheus", "kill", "-HUP", "1"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Force a real Prometheus alert for demo purposes.")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds to wait for alert firing")
    args = parser.parse_args()

    print("== Demo: forcing a Prometheus APIDown alert ==")
    print("Action: add one unreachable target under the same job, then wait for the alert rule.")
    print()

    try:
        backup_and_inject_dead_target()
        print("[OK] Injected dead target into the sentiment-api job and reloaded Prometheus.")

        deadline = time.monotonic() + args.timeout
        last_alerts: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            alerts = fetch_alerts()
            last_alerts = alerts
            firing = [
                item
                for item in alerts
                if item.get("labels", {}).get("alertname") == "APIDown"
                and item.get("state") == "firing"
            ]
            if firing:
                print()
                print("[ALERT] APIDown fired in Prometheus:")
                print(json.dumps(firing, indent=2))
                return 0
            time.sleep(5)

        print()
        print("[FAIL] APIDown did not fire before timeout.")
        if last_alerts:
            print(json.dumps(last_alerts, indent=2))
        return 1
    finally:
        print("\nRestoring Prometheus config...")
        restore_original_config()
        try:
            wait_for_ready()
            print("[OK] API still ready after restoring the config.")
        except Exception as exc:  # pragma: no cover - demo helper
            print(f"[WARN] Ready check failed after restore: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
