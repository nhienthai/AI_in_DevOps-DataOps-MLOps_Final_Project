#!/usr/bin/env python3
"""Run the identity-pair fairness probe against a running API.

The probe reaches the service over HTTP rather than importing the model, so the
number it produces describes the deployed system: its preprocessing, its
truncation, and the exact checkpoint that is answering requests.

Exits non-zero when the worst identity-pair gap exceeds the threshold, so this
can be used as a gate in CI or before a promotion:

    python scripts/run_fairness_probe.py --base-url http://localhost:8000
    python scripts/run_fairness_probe.py --threshold 0.05 --output-dir reports
    python scripts/run_fairness_probe.py --log-to-mlflow --run-id <RUN_ID>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sentiment.responsible.fairness import probe_over_http, write_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("fairness-probe")

DEFAULT_THRESHOLD = 0.10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="maximum tolerated identity-pair delta",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="where to write fairness_probe.json",
    )
    parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        help="log metrics and the report to an MLflow run",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="existing MLflow run to attach to; a new run is started when omitted",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI; falls back to the MLFLOW_TRACKING_URI environment",
    )
    return parser.parse_args(argv)


def _log_to_mlflow(
    result_metrics: dict[str, float],
    report_path: Path,
    run_id: str | None,
    tracking_uri: str | None,
) -> None:
    """Attach probe metrics and the report to an MLflow run."""
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(result_metrics)
            mlflow.log_artifact(str(report_path), artifact_path="fairness")
    else:
        with mlflow.start_run(run_name="fairness-probe"):
            mlflow.log_metrics(result_metrics)
            mlflow.log_artifact(str(report_path), artifact_path="fairness")


def main(argv: list[str] | None = None) -> int:
    """Run the probe and return a process exit code."""
    args = parse_args(argv)

    logger.info("probing %s", args.base_url)
    try:
        result = probe_over_http(args.base_url, batch_size=args.batch_size)
    except Exception as exc:
        logger.error("probe failed: %s", exc)
        return 2

    report_path = write_report(result, args.output_dir / "fairness_probe.json")

    logger.info("")
    logger.info("sentences        %d", result.n_sentences)
    logger.info("identity pairs   %d", result.n_pairs)
    logger.info("max delta        %.6f", result.max_delta)
    logger.info("mean delta       %.6f", result.mean_delta)
    logger.info("")
    logger.info("worst gap by dimension")
    for dimension, value in sorted(result.max_delta_by_dimension.items()):
        logger.info("  %-12s %.6f", dimension, value)
    logger.info("")
    logger.info("mean score by group")
    for group, value in sorted(result.group_mean_scores.items()):
        logger.info("  %-12s %.6f", group, value)
    logger.info("")
    logger.info("worst pairs")
    for pair in result.worst_pairs[:5]:
        logger.info(
            "  %.6f  %-10s %s vs %s  %s",
            pair["delta"],
            pair["dimension"],
            pair["group_a"],
            pair["group_b"],
            pair["template"],
        )
    logger.info("")
    logger.info("report written to %s", report_path)

    if args.log_to_mlflow:
        try:
            _log_to_mlflow(result.as_metrics(), report_path, args.run_id, args.tracking_uri)
            logger.info("metrics logged to MLflow")
        except Exception as exc:
            logger.error("MLflow logging failed: %s", exc)
            return 2

    passed = result.passes(args.threshold)
    logger.info("")
    logger.info(
        "gate: %s (max delta %.6f, threshold %.6f)",
        "PASS" if passed else "FAIL",
        result.max_delta,
        args.threshold,
    )
    if not passed:
        print(json.dumps({"fairness_max_delta": result.max_delta}), file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
