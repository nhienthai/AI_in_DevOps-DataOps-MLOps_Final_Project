"""Statistics behind the latency benchmark.

The measurement itself needs a running stack, but the reduction from samples to
reported numbers does not — and that reduction is where a benchmark quietly lies:
counting shed requests as fast responses, or averaging warm-up into p99.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "latency_bench.py"
_spec = importlib.util.spec_from_file_location("latency_bench", _SCRIPT)
assert _spec and _spec.loader
bench = importlib.util.module_from_spec(_spec)
sys.modules["latency_bench"] = bench
_spec.loader.exec_module(bench)


def test_percentile_uses_nearest_rank() -> None:
    ordered = [float(v) for v in range(1, 101)]
    assert bench.percentile(ordered, 0.50) == 51.0
    assert bench.percentile(ordered, 0.95) == 96.0
    assert bench.percentile(ordered, 0.99) == 100.0


def test_percentile_of_nothing_is_zero() -> None:
    assert bench.percentile([], 0.95) == 0.0


def test_percentile_never_runs_off_the_end() -> None:
    assert bench.percentile([7.0], 0.99) == 7.0


def _cell(**kwargs: Any) -> Any:
    return bench.Cell(concurrency=kwargs.pop("concurrency", 1), batch_size=kwargs.pop("batch", 1))


def test_shed_requests_are_not_counted_as_latency_samples() -> None:
    """A 429 answered in 0 ms must not flatter the percentiles."""
    cell = _cell()
    cell.record(200, 100.0, 90.0)
    for _ in range(9):
        cell.record(429, 0.2, None)

    summary = cell.summary()

    assert summary["ok"] == 1
    assert summary["shed"] == 9
    assert summary["shed_pct"] == pytest.approx(90.0)
    assert summary["p50_ms"] == pytest.approx(100.0)


def test_queue_overhead_is_client_time_minus_server_time() -> None:
    cell = _cell()
    for _ in range(100):
        cell.record(200, 120.0, 95.0)

    summary = cell.summary()

    assert summary["server_p95_ms"] == pytest.approx(95.0)
    assert summary["queue_overhead_p95_ms"] == pytest.approx(25.0)


def test_throughput_counts_predictions_not_just_requests() -> None:
    """A batch of 32 is one request and thirty-two predictions."""
    cell = _cell(batch=32)
    for _ in range(10):
        cell.record(200, 50.0, 45.0)
    cell.elapsed = 2.0

    summary = cell.summary()

    assert summary["requests_per_s"] == pytest.approx(5.0)
    assert summary["predictions_per_s"] == pytest.approx(160.0)


def test_transport_failures_are_visible_not_silent() -> None:
    cell = _cell()
    cell.record(200, 80.0, 70.0)
    cell.record(-1, 0.0, None)

    summary = cell.summary()

    assert summary["requests"] == 2
    assert summary["ok"] == 1


def test_corpus_falls_back_when_the_evaluation_csv_is_absent(tmp_path: Path) -> None:
    texts = bench.load_corpus(tmp_path / "absent.csv", seed=1)
    assert len(texts) == len(bench.FALLBACK_TEXTS)
    assert set(texts) == set(bench.FALLBACK_TEXTS)


def test_corpus_shuffle_is_deterministic(tmp_path: Path) -> None:
    corpus = tmp_path / "eval.csv"
    corpus.write_text(
        "text,actual\n" + "".join(f"câu số {i},POSITIVE\n" for i in range(50)), encoding="utf-8"
    )
    assert bench.load_corpus(corpus, seed=7) == bench.load_corpus(corpus, seed=7)
    assert bench.load_corpus(corpus, seed=7) != bench.load_corpus(corpus, seed=8)


def test_budget_flag_parses_and_defaults_to_the_slo() -> None:
    assert bench.parse_args([]).budget_ms == 200.0
    assert bench.parse_args(["--budget-ms", "0"]).budget_ms == 0.0
    assert bench.parse_args(["--concurrency", "1", "4"]).concurrency == [1, 4]
