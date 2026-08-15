#!/usr/bin/env python3
"""Generate traffic so the dashboards and alerts have something to show.

Five scenarios, each shaped to exercise a different panel or alert rather than
just to make the request counter move:

``steady``
    Mixed single and batch predictions at a fixed rate. Fills the request-rate,
    latency and class-distribution panels.
``burst``
    High concurrency against a pool sized ``max_concurrent_inferences``. Drives
    queue depth, queue latency and ``ml_inference_overloads_total``, and is the
    cheapest way to make ``HighLatencyP95`` fire.
``drift``
    Inputs far longer than the training reference. Moves ``ml_drift_psi`` toward
    and past 0.2, which fires ``DriftDetected`` after ten minutes.
``skew``
    One-sided sentiment. Starves the other two classes, which fires
    ``PredictionClassCollapse`` after fifteen minutes, and moves the positive
    share away from the service's 6h baseline, which fires ``PredictionSkew``.
    Run ``steady`` first, or use ``--scenario all``: both rules compare against
    measured history rather than a fixed prior, so on a stack that has only just
    started there is no baseline to deviate from and ``PredictionSkew`` stays
    quiet. ``PredictionClassCollapse`` fires either way.
``errors``
    Deliberately invalid requests. Fills the 4xx band and the per-``error_type``
    panel without touching the 5xx rate, which is the distinction
    ``HighErrorRate`` is built around.

Examples::

    python scripts/load_test.py --scenario steady --duration 300 --rps 20
    python scripts/load_test.py --scenario burst --duration 120 --concurrency 64
    python scripts/load_test.py --scenario all --duration 900
"""

from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx

POSITIVE_TEXTS = [
    "Giáo viên dạy rất hay và nhiệt tình.",
    "Bài giảng dễ hiểu, ví dụ sinh động.",
    "Thầy luôn giải đáp thắc mắc rất kỹ.",
    "Môn học bổ ích, em học được nhiều điều.",
    "Cách truyền đạt rõ ràng và cuốn hút.",
]

NEGATIVE_TEXTS = [
    "Bài giảng nhàm chán và khó theo dõi.",
    "Giáo viên đi quá nhanh, em không hiểu gì.",
    "Tài liệu sơ sài, không có ví dụ.",
    "Môn học nặng mà không thực tế.",
    "Cách chấm điểm không rõ ràng.",
]

NEUTRAL_TEXTS = [
    "Phòng học bình thường.",
    "Môn học có ba tín chỉ.",
    "Lịch học vào thứ hai và thứ tư.",
    "Giáo trình được phát đầu kỳ.",
    "Lớp có khoảng bốn mươi sinh viên.",
]

ALL_TEXTS = POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS

SCENARIOS = ("steady", "burst", "drift", "skew", "errors", "all")


@dataclass
class Stats:
    """Accumulated results for one scenario run."""

    statuses: Counter[int] = field(default_factory=Counter)
    latencies: list[float] = field(default_factory=list)
    failures: Counter[str] = field(default_factory=Counter)
    started: float = field(default_factory=time.perf_counter)

    def record(self, status: int, seconds: float) -> None:
        """Record one completed response."""
        self.statuses[status] += 1
        self.latencies.append(seconds)

    def record_failure(self, reason: str) -> None:
        """Record a request that never produced a response."""
        self.failures[reason] += 1

    @property
    def total(self) -> int:
        """Number of responses received."""
        return sum(self.statuses.values())

    def percentile(self, fraction: float) -> float:
        """Return the latency at ``fraction`` through the sorted samples."""
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        index = min(int(fraction * len(ordered)), len(ordered) - 1)
        return ordered[index]

    def report(self, scenario: str) -> None:
        """Print a human-readable summary."""
        elapsed = time.perf_counter() - self.started
        print(f"\n  {scenario}")
        print(f"    duration       {elapsed:.1f}s")
        print(f"    responses      {self.total} ({self.total / max(elapsed, 1e-9):.1f}/s)")
        if self.statuses:
            spread = ", ".join(f"{code}:{count}" for code, count in sorted(self.statuses.items()))
            print(f"    statuses       {spread}")
        if self.latencies:
            print(f"    latency mean   {statistics.fmean(self.latencies) * 1000:.1f} ms")
            print(f"    latency p95    {self.percentile(0.95) * 1000:.1f} ms")
            print(f"    latency p99    {self.percentile(0.99) * 1000:.1f} ms")
        if self.failures:
            spread = ", ".join(f"{name}:{count}" for name, count in self.failures.most_common())
            print(f"    failures       {spread}")


async def _post(
    client: httpx.AsyncClient, path: str, payload: dict[str, object], stats: Stats
) -> None:
    """Send one request and record its outcome."""
    started = time.perf_counter()
    try:
        response = await client.post(path, json=payload)
    except httpx.HTTPError as exc:
        stats.record_failure(type(exc).__name__)
        return
    stats.record(response.status_code, time.perf_counter() - started)


def _payload_for(scenario: str, rng: random.Random) -> tuple[str, dict[str, object]]:
    """Return the endpoint and body for one request in ``scenario``."""
    if scenario == "drift":
        filler = " ".join(rng.choice(ALL_TEXTS) for _ in range(rng.randint(20, 60)))
        return "/api/v1/predict", {"text": filler}

    if scenario == "skew":
        return "/api/v1/predict", {"text": rng.choice(POSITIVE_TEXTS)}

    if scenario == "errors":
        choice = rng.random()
        if choice < 0.4:
            return "/api/v1/predict", {"text": "   "}
        if choice < 0.7:
            return "/api/v1/predict", {"text": "x" * 6000}
        return "/api/v1/predict/batch", {"texts": ["ok"] * 128}

    if scenario == "burst":
        return "/api/v1/predict/batch", {
            "texts": [rng.choice(ALL_TEXTS) for _ in range(rng.randint(16, 48))]
        }

    if rng.random() < 0.3:
        return "/api/v1/predict/batch", {
            "texts": [rng.choice(ALL_TEXTS) for _ in range(rng.randint(2, 8))]
        }
    return "/api/v1/predict", {"text": rng.choice(ALL_TEXTS)}


async def run_scenario(
    base_url: str, scenario: str, duration: float, rps: float, concurrency: int, seed: int
) -> Stats:
    """Drive one scenario for ``duration`` seconds and return its statistics."""
    rng = random.Random(seed)
    stats = Stats()
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    deadline = time.perf_counter() + duration
    interval = 1.0 / rps if rps > 0 else 0.0
    pending: set[asyncio.Task[None]] = set()

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0, limits=limits) as client:
        while time.perf_counter() < deadline:
            while len(pending) >= concurrency:
                _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            path, payload = _payload_for(scenario, rng)
            pending.add(asyncio.create_task(_post(client, path, payload, stats)))
            if interval:
                await asyncio.sleep(interval)
        if pending:
            await asyncio.wait(pending)

    return stats


async def _preflight(base_url: str) -> bool:
    """Return whether the service is reachable and holding a model."""
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            response = await client.get("/ready")
    except httpx.HTTPError as exc:
        print(f"cannot reach {base_url}: {exc}", file=sys.stderr)
        return False
    if response.status_code != 200:
        print(f"{base_url}/ready returned {response.status_code}: {response.text}", file=sys.stderr)
        return False
    print(f"target {base_url} ready: {response.json()}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--scenario", choices=SCENARIOS, default="steady")
    parser.add_argument("--duration", type=float, default=120.0, help="seconds per scenario")
    parser.add_argument(
        "--rps", type=float, default=20.0, help="target requests/sec, 0 for unthrottled"
    )
    parser.add_argument("--concurrency", type=int, default=16, help="maximum in-flight requests")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> int:
    """Run the requested scenarios and report."""
    if not await _preflight(args.base_url):
        return 1

    scenarios = [s for s in SCENARIOS if s != "all"] if args.scenario == "all" else [args.scenario]
    if args.scenario == "all":
        per_scenario = args.duration / len(scenarios)
        print(f"running {len(scenarios)} scenarios, {per_scenario:.0f}s each")
    else:
        per_scenario = args.duration

    print("results")
    for scenario in scenarios:
        rps = 0.0 if scenario == "burst" else args.rps
        concurrency = max(args.concurrency, 32) if scenario == "burst" else args.concurrency
        stats = await run_scenario(
            args.base_url, scenario, per_scenario, rps, concurrency, args.seed
        )
        stats.report(scenario)

    print(
        "\ndone. Grafana: Sentiment — System & API, Model & Predictions, Fairness & Explainability"
    )
    return 0


def main() -> int:
    """Entry point."""
    return asyncio.run(_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
