"""Application configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable settings, sourced from ``SENTIMENT_``-prefixed variables."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env", extra="ignore")

    dataset_name: str = "amazon_polarity"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    train_size: int = Field(default=200_000, gt=0)
    val_size: int = Field(default=25_000, gt=0)
    test_size: int = Field(default=25_000, gt=0)
    random_seed: int = 42

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65_535)
    max_text_length: int = Field(default=5_000, gt=0)
    max_batch_size: int = Field(default=64, gt=0, le=1_000)
    low_confidence_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    drift_window_size: int = Field(default=1_000, ge=30)

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sentiment-amazon-polarity"
    model_registry_name: str = "sentiment-service-xlm-roberta"
    model_stage: str = "Production"
    predictor_backend: Literal["stub", "registry"] = "stub"

    model_name: str = "xlm-roberta-base"
    num_labels: int = Field(default=3, ge=2)
    label_map: dict[int, str] = Field(
        default_factory=lambda: {0: "negative", 1: "neutral", 2: "positive"}
    )
    rev_label_map: dict[str, int] = Field(
        default_factory=lambda: {"negative": 0, "neutral": 1, "positive": 2}
    )
    max_length: int = Field(default=256, gt=0)
    batch_size: int = Field(default=16, gt=0)
    epochs: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    artifacts_dir: Path = Path("artifacts")

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
