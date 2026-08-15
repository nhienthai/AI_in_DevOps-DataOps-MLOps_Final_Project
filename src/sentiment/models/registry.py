"""MLflow registry promotion and production predictor loading."""

import json
import logging
from pathlib import Path
from typing import Any

from sentiment.config import settings
from sentiment.serving.metrics import DriftReference
from sentiment.serving.predictor import Predictor

logger = logging.getLogger(__name__)


def register_and_promote_model(
    run_id: str,
    artifact_path: str = "model",
    model_name: str = settings.model_registry_name,
    stage: str = "Production",
    archive_existing: bool = True,
    tracking_uri: str | None = None,
) -> str:
    """Register a model run and promote the resulting version.

    Args:
        run_id: The MLflow run holding the artifact.
        artifact_path: Artifact subdirectory within the run.
        model_name: Registered model name.
        stage: Target stage.
        archive_existing: Archive whatever currently occupies ``stage``.
        tracking_uri: Overrides the configured URI. The default points at the
            Compose-internal ``http://mlflow:5000``, which does not resolve from
            the host, so anything promoting from a laptop must pass this.

    Returns:
        The new model version, as a string.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri or settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )
    return str(model_version.version)


def _latest_version(client: Any, model_name: str, stage: str) -> Any:
    versions = client.get_latest_versions(model_name, stages=[stage])
    if not versions:
        raise RuntimeError(f"No {stage} version exists for registered model '{model_name}'.")
    return max(versions, key=lambda item: int(item.version))


def _find_transformer_directory(artifact_root: Path) -> Path:
    if (artifact_root / "config.json").is_file():
        return artifact_root
    candidates = sorted(path.parent for path in artifact_root.rglob("config.json"))
    if not candidates:
        raise RuntimeError("Registered model artifact does not contain transformer config.json.")
    return candidates[0]


def _load_drift_reference(artifact_root: Path) -> DriftReference | None:
    candidates = list(artifact_root.rglob("drift_reference.json"))
    if not candidates:
        return None
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    return DriftReference(
        length_bin_edges=tuple(float(value) for value in payload["length_bin_edges"]),
        length_bin_freqs=tuple(float(value) for value in payload["length_bin_freqs"]),
        positive_prior=float(payload["positive_prior"]),
    )


def load_production_predictor(
    *,
    tracking_uri: str,
    stage: str = "Production",
    model_name: str = settings.model_registry_name,
) -> Predictor:
    """Download the promoted artifact and expose its serving metadata.

    Dispatches on what the run actually logged: a ``*.joblib`` file is the
    TF-IDF baseline, anything else is a Hugging Face directory. The baseline
    matters beyond being a fallback — it is the only model that trains and serves
    on CPU in seconds, so it is what makes ``predictor_backend=registry``
    demonstrable without a GPU.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    from sentiment.models.transformer import TransformerPredictor

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version = _latest_version(client, model_name, stage)
    run = client.get_run(version.run_id)
    artifact_root = Path(client.download_artifacts(version.run_id, "model"))

    # Typed Any rather than Predictor: the protocol declares only `version` and
    # `predict`, while the concrete classes also carry the serving metadata set
    # below. Widening the protocol to cover metadata would force every predictor
    # to implement fields the serving layer already reads defensively.
    predictor: Any
    joblib_files = sorted(artifact_root.glob("*.joblib"))
    if joblib_files:
        from sentiment.models.baseline import BaselinePredictor

        predictor = BaselinePredictor.load(str(joblib_files[0]), model_version=str(version.version))
    else:
        model_directory = _find_transformer_directory(artifact_root)
        predictor = TransformerPredictor.from_pretrained(
            str(model_directory), model_version=str(version.version)
        )
    predictor.stage = stage
    predictor.run_id = str(version.run_id)
    predictor.trained_at = run.data.tags.get("trained_at")
    predictor.metrics = {key: float(value) for key, value in run.data.metrics.items()}
    fairness = run.data.metrics.get("fairness_max_delta")
    predictor.fairness_delta = float(fairness) if fairness is not None else None
    drift_reference = _load_drift_reference(artifact_root)
    if drift_reference is not None:
        setattr(predictor, "drift_reference", drift_reference)
    logger.info(
        "Loaded model %s version %s from stage %s",
        model_name,
        version.version,
        stage,
    )
    if not isinstance(predictor, Predictor):
        raise RuntimeError(
            f"Registered model {model_name} v{version.version} does not satisfy the "
            "Predictor protocol."
        )
    return predictor


def load_production_model(model_name: str = settings.model_registry_name) -> Predictor:
    """Backward-compatible wrapper around the serving predictor loader."""
    return load_production_predictor(
        tracking_uri=settings.mlflow_tracking_uri,
        stage=settings.model_stage,
        model_name=model_name,
    )
