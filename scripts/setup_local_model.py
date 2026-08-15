#!/usr/bin/env python3
"""Unpack donated model weights and the MLflow tracking database.

The archives arrive out of band because none of it belongs in git: the weights
run to gigabytes and ``models/`` is ignored. This script turns them into the
directories the serving layer expects, and it extracts only what inference needs.

Both archives carry training checkpoints holding ``optimizer.pt`` state — 6.6 GB
in the XLM-R bundle, 2.1 GB in the PhoBERT one — which matter only for resuming a
fine-tune. Taking the whole archive would cost gigabytes of disk to serve a model
that is a fraction of the size, so the model root is unpacked and the checkpoint
directories are skipped.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sentiment.models.donated import read_donated_run  # noqa: E402


@dataclass(frozen=True)
class Bundle:
    """One donated model archive and how its contents must be served."""

    name: str
    archive: str
    directory: str
    description: str
    # Written into serving_metadata.json; the local backend turns this into the
    # InputFormat it applies before tokenizing.
    preprocessing: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_source: str = ""


BUNDLES = (
    Bundle(
        name="xlm-roberta",
        archive="model_weights.zip",
        directory="artifacts/xlm-roberta",
        description="XLM-RoBERTa base fine-tuned on raw UIT-VSFC sentences",
    ),
    Bundle(
        name="phobert",
        archive="phobert_model_weights.zip",
        directory="artifacts/phobert-sota",
        description="PhoBERT-v2 fine-tuned with focal loss on topic-injected input",
        # notebooks/phobert-v2.ipynb trains on "Chủ đề: {topic} | {cleaned}", never
        # on a bare sentence. The HTTP contract carries no topic, so serving pins
        # the dataset's own default value; measured on the 3166-example test set
        # that scores marginally better than passing the true topic.
        preprocessing={"clean_dataset_artifacts": True, "template": "Chủ đề: others | {text}"},
        metrics={"test_accuracy": 0.9416, "test_macro_f1": 0.8457},
        metrics_source=(
            "measured on the 3166-example UIT-VSFC test split through this exact "
            "serving input format"
        ),
    ),
)


def _model_root(names: list[str]) -> str:
    """Locate the directory inside the archive that holds the servable model.

    Chosen by finding ``config.json`` outside any ``checkpoint-*`` directory, so a
    renamed bundle keeps working without editing this script.
    """
    candidates = [
        name.rsplit("config.json", 1)[0]
        for name in names
        if name.endswith("config.json") and "checkpoint-" not in name
    ]
    if not candidates:
        raise SystemExit("archive contains no config.json outside a checkpoint directory")
    return min(candidates, key=len)


def _extract_model(archive: Path, destination: Path) -> list[Path]:
    """Copy the model root out of the archive, leaving checkpoints behind."""
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        root = _model_root(names)
        members = [
            name
            for name in names
            if name.startswith(root) and name != root and "/" not in name[len(root) :]
        ]
        for name in members:
            target = destination / name[len(root) :]
            if target.exists() and target.stat().st_size > 0:
                print(f"  skip   {target.name}")
                written.append(target)
                continue
            print(f"  write  {target.name}")
            with bundle.open(name) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            written.append(target)
    if not any(path.name == "config.json" for path in written):
        raise SystemExit(f"{archive.name} yielded no config.json")
    return written


def _extract_tracking_db(archive: Path, destination: Path) -> Path:
    """Expand the MLflow archive, which is small enough to take whole."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    database = destination / "mlflow.db"
    if not database.is_file():
        raise SystemExit(f"{archive.name} did not contain mlflow.db")
    return database


def _write_metadata(bundle: Bundle, directory: Path, database: Path | None) -> dict[str, Any]:
    """Record lineage next to the weights for the local backend to read."""
    metadata: dict[str, Any] = {
        "model": bundle.name,
        "description": bundle.description,
        "metrics": dict(bundle.metrics),
    }
    if bundle.preprocessing:
        metadata["preprocessing"] = bundle.preprocessing
    if bundle.metrics_source:
        metadata["metrics_source"] = bundle.metrics_source

    # Only the XLM-R bundle came with a tracking database; PhoBERT's provenance is
    # its notebook, so its metadata carries measurements rather than a run id.
    if database is not None and not bundle.metrics:
        run = read_donated_run(database)
        metadata.update(
            {
                "run_id": run.run_id,
                "run_name": run.run_name,
                "experiment_name": run.experiment_name,
                "start_time_ms": run.start_time_ms,
                "params": run.params,
                "metrics": run.metrics,
                "source_git_commit": run.tags.get("mlflow.source.git.commit"),
            }
        )
    (directory / "serving_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=[bundle.name for bundle in BUNDLES],
        help="Unpack a single bundle instead of every archive present",
    )
    parser.add_argument(
        "--archives-dir", type=Path, default=REPO_ROOT / "models", help="Where the zips live"
    )
    parser.add_argument(
        "--tracking-dir",
        type=Path,
        default=REPO_ROOT / "models" / "donated",
        help="Where the donated MLflow database is expanded",
    )
    args = parser.parse_args()

    tracking_archive = args.archives_dir / "mlflow_db.zip"
    database: Path | None = None
    if tracking_archive.is_file():
        print(f"=== Unpacking {tracking_archive.name} -> {args.tracking_dir} ===")
        database = _extract_tracking_db(tracking_archive, args.tracking_dir)
    else:
        print(f"[WARN] {tracking_archive} not found; models will carry no run lineage")

    wanted = [b for b in BUNDLES if args.only is None or b.name == args.only]
    unpacked = 0
    for bundle in wanted:
        archive = args.archives_dir / bundle.archive
        if not archive.is_file():
            print(f"\n[SKIP] {bundle.name}: {archive} not found")
            continue
        directory = REPO_ROOT / bundle.directory
        print(f"\n=== Unpacking {bundle.archive} -> {directory} ===")
        written = _extract_model(archive, directory)
        metadata = _write_metadata(bundle, directory, database)
        size = sum(path.stat().st_size for path in written)
        print(f"  {len(written)} files, {size / 1024 ** 3:.2f} GiB")
        metrics = metadata.get("metrics", {})
        for key in ("test_accuracy", "test_macro_f1"):
            if key in metrics:
                print(f"  {key:<16} {metrics[key]:.4f}")
        if "preprocessing" in metadata:
            print(f"  input format     {metadata['preprocessing']['template']}")
        unpacked += 1

    if unpacked == 0:
        print("\n[FAIL] no archives were unpacked", file=sys.stderr)
        return 1
    print("\nNext: python scripts/import_donated_run.py   (logs the XLM-R run into MLflow)")
    print("Serve with SENTIMENT_PREDICTOR_BACKEND=local and SENTIMENT_LOCAL_MODEL_DIR=<dir>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
