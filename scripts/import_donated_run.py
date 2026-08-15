#!/usr/bin/env python3
"""Replay the donated Kaggle training run into this project's MLflow.

The fine-tune happened elsewhere, so its history lives in a SQLite file rather
than in the tracking server the stack runs. This script reads that file and logs
an equivalent run — params, metrics, source tags, and the per-example evaluation
CSV — then registers a model version and promotes it to Production.

Weights are deliberately not uploaded. They are 1.1 GB, they already sit in
``artifacts/xlm-roberta`` after ``setup_local_model.py``, and the API serves them
from there via ``SENTIMENT_PREDICTOR_BACKEND=local``. The registered version
records where they are so the registry entry and the running model agree; it is
lineage, not a distribution channel.

Re-running is safe: a run already carrying this ``donated_run_id`` tag is left
alone unless ``--force`` is passed.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sentiment.config import settings  # noqa: E402
from sentiment.models.donated import DonatedRun, read_donated_run  # noqa: E402

DONATED_RUN_TAG = "donated_run_id"
WEIGHTS_POINTER_DIR = "weights_pointer"


def _run_from_model_metadata(model_dir: Path) -> DonatedRun:
    """Build a run from ``serving_metadata.json`` for a model with no database.

    The PhoBERT bundle arrived as weights and a notebook, without the tracking
    database the XLM-R run came with. Its identity is therefore derived from the
    metadata itself: a stable hash keeps re-imports idempotent the same way a real
    run id does, and the weights' mtime stands in for a training timestamp.
    """
    import hashlib

    path = model_dir / "serving_metadata.json"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run scripts/setup_local_model.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    weights = model_dir / "model.safetensors"
    started_ms = int(weights.stat().st_mtime * 1000) if weights.is_file() else 0
    name = str(payload.get("model", model_dir.name))
    params = {str(k): str(v) for k, v in (payload.get("params") or {}).items()}
    preprocessing = payload.get("preprocessing") or {}
    if preprocessing:
        params["input_template"] = str(preprocessing.get("template"))
        params["clean_dataset_artifacts"] = str(preprocessing.get("clean_dataset_artifacts"))
    return DonatedRun(
        run_id=f"{name}-{digest[:12]}",
        run_name=f"fine-tune-{name}",
        experiment_name="sentiment-analysis-uit-vsfc",
        start_time_ms=started_ms,
        params=params,
        metrics={str(k): float(v) for k, v in (payload.get("metrics") or {}).items()},
        tags={
            "metrics_source": str(payload.get("metrics_source", "")),
            "description": str(payload.get("description", "")),
        },
    )


def _log_weights_pointer(mlflow: Any, model_dir: Path, run: DonatedRun) -> None:
    """Record where the weights live, since they are not uploaded."""
    files = (
        {path.name: path.stat().st_size for path in sorted(model_dir.iterdir()) if path.is_file()}
        if model_dir.is_dir()
        else {}
    )
    pointer = {
        "weights_are_uploaded": False,
        "weights_dir": str(model_dir),
        "serving_backend": "local",
        "donated_run_id": run.run_id,
        "files": files,
        "restore_with": "python scripts/setup_local_model.py",
    }
    with tempfile.TemporaryDirectory() as staging:
        target = Path(staging) / "local_model_reference.json"
        target.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
        mlflow.log_artifact(str(target), artifact_path=WEIGHTS_POINTER_DIR)


def _existing_run(client: Any, experiment_id: str, run: DonatedRun) -> str | None:
    """Return the id of a previous import of this run, if there is one."""
    matches = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.{DONATED_RUN_TAG} = '{run.run_id}'",
        max_results=1,
    )
    return str(matches[0].info.run_id) if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=REPO_ROOT / "models" / "donated" / "mlflow.db",
        help="Donated MLflow SQLite database",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=REPO_ROOT / "models" / "donated" / "artifacts" / "eval_results.csv",
        help="Per-example evaluation output logged alongside the run",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "xlm-roberta",
        help="Where the weights this version refers to actually live",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5001",
        help="Target MLflow server. The Compose-internal http://mlflow:5000 does "
        "not resolve from the host, so the default is the published port.",
    )
    parser.add_argument("--run-id", default=None, help="Donated run to import (default: latest)")
    parser.add_argument(
        "--from-model-metadata",
        action="store_true",
        help="Take the run from --model-dir's serving_metadata.json instead of the "
        "tracking database. Use for bundles that arrived without one, like PhoBERT.",
    )
    parser.add_argument(
        "--stage", default="Production", help="Registry stage to promote the version into"
    )
    parser.add_argument(
        "--force", action="store_true", help="Import again even if this run was already imported"
    )
    args = parser.parse_args()
    args.eval_results_given = any(a.startswith("--eval-results") for a in sys.argv[1:])

    import mlflow
    from mlflow.tracking import MlflowClient

    if args.from_model_metadata:
        run = _run_from_model_metadata(args.model_dir)
    else:
        run = read_donated_run(args.database, run_id=args.run_id)
    print(f"=== Importing {run.run_name} ({run.run_id}) ===")
    print(f"Target       : {args.tracking_uri}")
    print(f"Experiment   : {run.experiment_name}")

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    experiment_id = mlflow.set_experiment(run.experiment_name).experiment_id

    already = _existing_run(client, experiment_id, run)
    if already and not args.force:
        print(f"[SKIP] already imported as run {already}. Pass --force to import again.")
        return 0

    with mlflow.start_run(run_name=run.run_name) as active:
        mlflow.log_params(run.params)
        mlflow.log_metrics(run.metrics)
        mlflow.set_tags(
            {
                DONATED_RUN_TAG: run.run_id,
                "donated_from": "kaggle",
                "trained_at": run.trained_at,
                "weights_location": str(args.model_dir),
                "serving_backend": "local",
                # Carried over so the imported run still points at the code that
                # produced it, rather than at this import script.
                "mlflow.source.name": run.tags.get("mlflow.source.name", "unknown"),
                "mlflow.source.git.commit": run.tags.get("mlflow.source.git.commit", "unknown"),
                "mlflow.source.git.branch": run.tags.get("mlflow.source.git.branch", "unknown"),
            }
        )
        # The donated eval CSV holds the XLM-R run's own predictions, so it must
        # not be attached to a run imported from a different model's metadata.
        if args.from_model_metadata and not args.eval_results_given:
            print("Logged       : (no per-example evaluation for this bundle)")
        elif args.eval_results.is_file():
            mlflow.log_artifact(str(args.eval_results), artifact_path="evaluation")
            print(f"Logged       : {args.eval_results.name}")
        else:
            print(f"[WARN] {args.eval_results} not found; run imported without it")

        # The registry refuses a model version whose source sits outside the
        # run's own artifact directory, so the run carries a pointer describing
        # where the weights really are. Deliberately not an MLmodel directory:
        # this records provenance and must not look loadable.
        _log_weights_pointer(mlflow, args.model_dir, run)
        imported_run_id = active.info.run_id
        artifact_uri = active.info.artifact_uri

    for key in sorted(run.metrics):
        print(f"  {key:<26} {run.metrics[key]:.4f}")

    name = settings.model_registry_name
    try:
        client.create_registered_model(name)
    except Exception:  # noqa: BLE001 - already exists is the only expected case
        pass

    version = client.create_model_version(
        name=name,
        source=f"{artifact_uri}/{WEIGHTS_POINTER_DIR}",
        run_id=imported_run_id,
        tags={
            "weights_location": "local",
            "weights_dir": str(args.model_dir),
            "donated_run_id": run.run_id,
        },
    )
    client.transition_model_version_stage(
        name=name, version=version.version, stage=args.stage, archive_existing_versions=True
    )

    print()
    print(f"Imported run : {imported_run_id}")
    print(f"Registered   : {name} v{version.version} -> {args.stage}")
    print(f"Weights      : {args.model_dir}")
    print()
    print("Serve it with SENTIMENT_PREDICTOR_BACKEND=local (weights were not uploaded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
