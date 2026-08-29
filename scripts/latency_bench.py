#!/usr/bin/env python3
"""Measure serving latency precisely enough to put the numbers in a report.

This is not ``load_test.py``. That script generates traffic so the dashboards and
alert rules have something to react to; its statistics are a by-product. This one
exists to answer "what is p95, and is it inside the budget", which needs things a
traffic generator does not do:

* **Warm-up is discarded.** The first requests after a reload carry model warm-up
  and allocator noise — a measured p99 of 324 ms collapsed to 104 ms once the
  first ten were dropped. Reporting them as steady-state latency is simply wrong.
* **Two clocks, not one.** Every response carries the server's own ``latency_ms``
  for the inference itself. Subtracting it from the wall time the client saw
  isolates queueing and HTTP overhead, which is what tells you whether to raise
  ``max_concurrent_inferences`` or to give the container more CPU.
* **A sweep, not a point.** Latency at one concurrency says nothing about where
  the service starts shedding. The grid walks batch size against concurrency and
  reports the 429 rate in every cell, so the shedding boundary is visible.

Examples::

    python scripts/latency_bench.py                       # full grid, budget check
    python scripts/latency_bench.py --budget-ms 200       # fail the shell if p95 blows it
    python scripts/latency_bench.py --json bench.json     # keep the numbers
    python scripts/latency_bench.py --concurrency 1 --batch-size 1 --requests 300
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

try:
    import httpx
except ImportError:  # pragma: no cover - the message is the whole point
    print("httpx is required: pip install httpx", file=sys.stderr)
    raise SystemExit(2) from None

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "models" / "donated" / "artifacts" / "eval_results.csv"

# Used when the donated evaluation corpus has not been unpacked. Real sentences
# matter: latency follows token count, and invented filler is not representative.
FALLBACK_TEXTS = (
    "giáo viên dạy rất hay và nhiệt tình",
    "môn học quá chán, tài liệu sơ sài",
    "lớp học bình thường",
    "bài giảng dễ hiểu nhưng hơi nhanh",
    "phòng học nóng, thiết bị cũ",
)


def load_corpus(path: Path, seed: int) -> list[str]:
    """Return real review sentences, shuffled deterministically."""
    texts: list[str]
    if path.is_file():
        with path.open(encoding="utf-8-sig") as handle:
            texts = [row["text"] for row in csv.DictReader(handle) if row.get("text")]
    else:
        texts = list(FALLBACK_TEXTS)
    random.Random(seed).shuffle(texts)
    return texts


def percentile(ordered: Sequence[float], fraction: float) -> float:
    """Return the value at ``fraction`` through an already-sorted sequence.

    Nearest-rank rather than interpolated: with a few hundred samples the
    interpolated value implies a precision the measurement does not have.
    """
    if not ordered:
        return 0.0
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


@dataclass
class Cell:
    """One (concurrency, batch size) point of the grid."""

    concurrency: int
    batch_size: int
    client_ms: list[float] = field(default_factory=list)
    server_ms: list[float] = field(default_factory=list)
    statuses: dict[int, int] = field(default_factory=dict)
    elapsed: float = 0.0

    def record(self, status: int, client_ms: float, server_ms: float | None) -> None:
        self.statuses[status] = self.statuses.get(status, 0) + 1
        if status == 200:
            self.client_ms.append(client_ms)
            if server_ms is not None:
                self.server_ms.append(server_ms)

    @property
    def ok(self) -> int:
        return self.statuses.get(200, 0)

    @property
    def shed(self) -> int:
        """Requests refused by admission control rather than served."""
        return self.statuses.get(429, 0) + self.statuses.get(503, 0)

    @property
    def total(self) -> int:
        return sum(self.statuses.values())

    def summary(self) -> dict[str, Any]:
        """Reduce the samples to the numbers worth reporting."""
        client = sorted(self.client_ms)
        server = sorted(self.server_ms)
        predictions = self.ok * self.batch_size
        overhead = percentile(client, 0.95) - percentile(server, 0.95) if client and server else 0.0
        return {
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "requests": self.total,
            "ok": self.ok,
            "shed": self.shed,
            "shed_pct": 100.0 * self.shed / self.total if self.total else 0.0,
            "p50_ms": percentile(client, 0.50),
            "p90_ms": percentile(client, 0.90),
            "p95_ms": percentile(client, 0.95),
            "p99_ms": percentile(client, 0.99),
            "max_ms": max(client) if client else 0.0,
            "mean_ms": statistics.fmean(client) if client else 0.0,
            "server_p95_ms": percentile(server, 0.95),
            "queue_overhead_p95_ms": overhead,
            "requests_per_s": self.ok / self.elapsed if self.elapsed else 0.0,
            "predictions_per_s": predictions / self.elapsed if self.elapsed else 0.0,
        }


async def _one_request(
    client: httpx.AsyncClient, texts: list[str], cell: Cell, semaphore: asyncio.Semaphore
) -> None:
    """Send one request and record both clocks."""
    single = len(texts) == 1
    path = "/api/v1/predict" if single else "/api/v1/predict/batch"
    payload: dict[str, Any] = {"text": texts[0]} if single else {"texts": texts}
    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(path, json=payload)
        except httpx.HTTPError:
            cell.record(-1, 0.0, None)
            return
        client_ms = (time.perf_counter() - started) * 1000.0
    server_ms: float | None = None
    if response.status_code == 200:
        body = response.json()
        if single:
            server_ms = float(body["latency_ms"])
        else:
            first = next(
                (item["prediction"] for item in body["results"] if item["prediction"]), None
            )
            server_ms = float(first["latency_ms"]) if first else None
    cell.record(response.status_code, client_ms, server_ms)


async def measure(
    base_url: str,
    corpus: list[str],
    concurrency: int,
    batch_size: int,
    requests: int,
    warmup: int,
    timeout: float,
) -> Cell:
    """Run one grid cell, discarding ``warmup`` requests before recording."""
    cell = Cell(concurrency=concurrency, batch_size=batch_size)
    discard = Cell(concurrency=concurrency, batch_size=batch_size)
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, limits=limits) as client:
        cursor = 0

        def next_batch() -> list[str]:
            nonlocal cursor
            batch = [corpus[(cursor + i) % len(corpus)] for i in range(batch_size)]
            cursor += batch_size
            return batch

        if warmup:
            await asyncio.gather(
                *(_one_request(client, next_batch(), discard, semaphore) for _ in range(warmup))
            )
        started = time.perf_counter()
        await asyncio.gather(
            *(_one_request(client, next_batch(), cell, semaphore) for _ in range(requests))
        )
        cell.elapsed = time.perf_counter() - started
    return cell


def print_table(rows: list[dict[str, Any]]) -> None:
    """Render the grid the way it is easiest to read aloud from."""
    header = (
        f"  {'conc':>4} {'batch':>5} {'n':>5} {'p50':>7} {'p95':>7} {'p99':>7} "
        f"{'max':>7} {'req/s':>7} {'pred/s':>8} {'shed':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(
            f"  {row['concurrency']:>4} {row['batch_size']:>5} {row['requests']:>5} "
            f"{row['p50_ms']:>7.1f} {row['p95_ms']:>7.1f} {row['p99_ms']:>7.1f} "
            f"{row['max_ms']:>7.1f} {row['requests_per_s']:>7.1f} "
            f"{row['predictions_per_s']:>8.1f} {row['shed_pct']:>5.0f}%"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Concurrency levels to sweep",
    )
    parser.add_argument(
        "--batch-size", type=int, nargs="+", default=[1, 8, 32], help="Batch sizes to sweep"
    )
    parser.add_argument("--requests", type=int, default=120, help="Recorded requests per cell")
    parser.add_argument("--warmup", type=int, default=15, help="Requests discarded per cell")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout")
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=200.0,
        help="p95 budget checked against the single-request cell; 0 disables the check",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the full results here")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    corpus = load_corpus(CORPUS, args.seed)
    print(f"=== Latency benchmark against {args.base_url} ===")
    print(f"  corpus {len(corpus)} sentences, {args.requests} requests/cell, {args.warmup} warm-up")

    rows: list[dict[str, Any]] = []
    for batch_size in args.batch_size:
        for concurrency in args.concurrency:
            cell = await measure(
                args.base_url,
                corpus,
                concurrency,
                batch_size,
                args.requests,
                args.warmup,
                args.timeout,
            )
            rows.append(cell.summary())
    print()
    print_table(rows)

    baseline = next(
        (r for r in rows if r["concurrency"] == 1 and r["batch_size"] == 1),
        rows[0] if rows else None,
    )
    if baseline is None:
        print("\n[FAIL] no measurements", file=sys.stderr)
        return 1

    print()
    print(f"  single request p50 {baseline['p50_ms']:.1f} ms · p95 {baseline['p95_ms']:.1f} ms")
    print(
        f"  of which inference {baseline['server_p95_ms']:.1f} ms; "
        f"queue + HTTP overhead {baseline['queue_overhead_p95_ms']:.1f} ms at p95"
    )
    shedding = [r for r in rows if r["shed_pct"] > 0]
    if shedding:
        first = min(shedding, key=lambda r: (r["concurrency"], r["batch_size"]))
        print(
            f"  admission control starts shedding at concurrency {first['concurrency']}, "
            f"batch {first['batch_size']} ({first['shed_pct']:.0f}% refused)"
        )
    else:
        print("  nothing was shed: the grid never reached the concurrency limit")

    if args.json:
        args.json.write_text(
            json.dumps({"base_url": args.base_url, "cells": rows}, indent=2), encoding="utf-8"
        )
        print(f"  wrote {args.json}")

    if args.budget_ms > 0:
        measured = baseline["p95_ms"]
        verdict = "PASS" if measured <= args.budget_ms else "FAIL"
        print(f"\n  [{verdict}] p95 {measured:.1f} ms against a {args.budget_ms:.0f} ms budget")
        return 0 if verdict == "PASS" else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
