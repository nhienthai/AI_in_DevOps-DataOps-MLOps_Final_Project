"""Serving a transformer straight from a local directory.

The registry backend is the production path, but it needs the promoted artifact
to travel through MLflow: 1.1 GB uploaded once and downloaded again by every API
container that starts. When the weights already sit on disk — as they do after
``scripts/setup_local_model.py`` unpacks the donated archive — this backend
skips that round trip and hands the same ``TransformerPredictor`` to the runtime.

Serving metadata is read from ``serving_metadata.json`` next to the weights so
``/model-info`` and the Prometheus gauges describe the model with the same
fields the registry path fills in from an MLflow run.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentiment.config import Settings
from sentiment.models.text_format import InputFormat
from sentiment.serving.predictor import Predictor

logger = logging.getLogger(__name__)

METADATA_FILENAME = "serving_metadata.json"


def _read_serving_metadata(model_dir: Path) -> dict[str, Any]:
    """Return the donated run's lineage, or empty defaults when it is absent.

    Metadata is a convenience, not a precondition: a directory copied in by hand
    still serves. Losing it costs dashboard labels, so a parse failure is logged
    rather than raised.
    """
    empty: dict[str, Any] = {
        "run_id": None,
        "trained_at": None,
        "metrics": {},
        "input_format": InputFormat(),
    }
    path = model_dir / METADATA_FILENAME
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return empty
    if not isinstance(payload, dict):
        logger.warning("Ignoring %s: expected a JSON object", path)
        return empty

    raw_metrics = payload.get("metrics")
    metrics: dict[str, float] = {}
    if isinstance(raw_metrics, dict):
        for key, value in raw_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            metrics[str(key)] = float(value)

    return {
        "run_id": payload.get("run_id"),
        "trained_at": _trained_at(payload.get("start_time_ms")),
        "metrics": metrics,
        "input_format": InputFormat.from_metadata(payload.get("preprocessing")),
    }


def _trained_at(start_time_ms: object) -> str | None:
    """Convert MLflow's epoch milliseconds into the ISO string serving reports."""
    if not isinstance(start_time_ms, (int, float)) or isinstance(start_time_ms, bool):
        return None
    return datetime.fromtimestamp(float(start_time_ms) / 1000.0, tz=timezone.utc).isoformat()


def _model_version(model_dir: Path, run_id: object) -> str:
    """Identify the model by its originating run, falling back to the directory."""
    if isinstance(run_id, str) and run_id:
        return f"local-{run_id[:8]}"
    return f"local-{model_dir.name}"


def load_local_predictor(settings: Settings) -> Predictor:
    """Load the transformer sitting in ``settings.local_model_dir``."""
    model_dir = Path(settings.local_model_dir)
    if not model_dir.is_dir():
        raise RuntimeError(
            f"Local model directory {model_dir} does not exist. "
            "Run scripts/setup_local_model.py to unpack the weights."
        )
    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"Local model directory {model_dir} has no config.json.")

    from sentiment.models.transformer import TransformerPredictor

    metadata = _read_serving_metadata(model_dir)
    version = _model_version(model_dir, metadata["run_id"])

    # Typed Any for the same reason registry.py does: the Predictor protocol
    # covers `version` and `predict`, while serving reads this metadata off the
    # concrete class defensively.
    input_format = metadata["input_format"]
    predictor: Any = TransformerPredictor.from_pretrained(
        str(model_dir), model_version=version, input_format=input_format
    )
    predictor.stage = settings.model_stage
    predictor.run_id = metadata["run_id"]
    predictor.trained_at = metadata["trained_at"]
    predictor.metrics = metadata["metrics"]
    fairness = metadata["metrics"].get("fairness_max_delta")
    predictor.fairness_delta = float(fairness) if fairness is not None else None

    logger.info(
        "Loaded local model %s from %s (input_format=%s)",
        version,
        model_dir,
        "identity" if input_format.is_identity else input_format,
    )
    if not isinstance(predictor, Predictor):
        raise RuntimeError(f"Local model at {model_dir} does not satisfy the Predictor protocol.")
    return predictor
