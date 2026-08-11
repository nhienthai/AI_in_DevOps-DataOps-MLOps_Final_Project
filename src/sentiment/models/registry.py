"""MLflow Model Registry integration helpers."""

import logging
from typing import Optional
import mlflow
from mlflow.tracking import MlflowClient

from src.sentiment.config import settings

logger = logging.getLogger(__name__)


def register_and_promote_model(
    run_id: str,
    artifact_path: str = "model",
    model_name: str = settings.model_registry_name,
    stage: str = "Production",
    archive_existing: bool = True,
) -> str:
    """Register a model run in MLflow Model Registry and transition it to target stage."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    logger.info("Registering model from URI: %s as '%s'", model_uri, model_name)
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)

    logger.info(
        "Transitioning model '%s' version %s to stage '%s'",
        model_name,
        model_version.version,
        stage,
    )
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=stage,
        archive_existing_versions=archive_existing,
    )

    return model_version.version


def load_production_model(model_name: str = settings.model_registry_name):
    """Load model currently in Production stage from MLflow Registry."""
    model_uri = f"models:/{model_name}/Production"
    logger.info("Loading Production model from MLflow: %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)
