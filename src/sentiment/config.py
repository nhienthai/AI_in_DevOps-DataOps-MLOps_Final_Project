"""Application configuration for sentiment service."""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Settings:
    # Dataset configuration
    dataset_name: str = os.getenv("DATASET_NAME", "tridm/UIT-VSFC")
    num_labels: int = 3
    label_map: Dict[int, str] = field(
        default_factory=lambda: {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    )
    rev_label_map: Dict[str, int] = field(
        default_factory=lambda: {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    )

    # Model configuration
    model_name: str = os.getenv("MODEL_NAME", "xlm-roberta-base")
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    epochs: int = int(os.getenv("EPOCHS", "3"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "2e-5"))

    # MLflow & Artifact configuration
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    model_registry_name: str = os.getenv(
        "MODEL_REGISTRY_NAME", "sentiment-service-xlm-roberta"
    )
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "./artifacts")


settings = Settings()
