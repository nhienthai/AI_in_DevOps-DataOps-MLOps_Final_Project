"""Application configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable settings, sourced from ``SENTIMENT_``-prefixed variables."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env", extra="ignore")

    dataset_name: str = "tridm/UIT-VSFC"
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
    max_request_body_bytes: int = Field(default=1_000_000, gt=0)
    max_concurrent_inferences: int = Field(default=2, gt=0, le=64)
    inference_timeout_seconds: float = Field(default=30.0, gt=0)
    queue_timeout_seconds: float = Field(default=1.0, gt=0)
    warmup_text: str = "Service warm-up review."
    low_confidence_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    drift_window_size: int = Field(default=1_000, ge=30)

    # Host-side default. Compose publishes MLflow on 5001 and overrides this with
    # the in-network http://mlflow:5000 for the services themselves.
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "sentiment-amazon-polarity"
    model_registry_name: str = "sentiment-service-xlm-roberta"
    model_stage: str = "Production"
    predictor_backend: Literal["stub", "registry", "local"] = "stub"
    # Where ``local`` reads weights from. scripts/setup_local_model.py writes here.
    local_model_dir: Path = Path("artifacts/xlm-roberta")
    reload_token: str | None = None
    build_revision: str = "unknown"

    model_name: str = "xlm-roberta-base"
    model_dataset_name: str = "tridm/UIT-VSFC"
    num_labels: Literal[3] = 3
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

    @field_validator("reload_token", mode="before")
    @classmethod
    def blank_reload_token_is_disabled(cls, value: object) -> object:
        """Treat an omitted or blank secret as a disabled reload endpoint."""
        return None if value == "" else value

    @model_validator(mode="after")
    def one_dataset_for_data_and_training(self) -> "Settings":
        """Reject a configuration where the data and training layers disagree.

        These were briefly two different datasets — the data layer ingesting
        ``amazon_polarity`` while training loaded UIT-VSFC — which made the
        quality gate guard data nobody trained on. Keeping both names is
        tolerable; letting them differ is not.
        """
        if self.dataset_name != self.model_dataset_name:
            raise ValueError(
                "dataset_name and model_dataset_name must match: "
                f"{self.dataset_name!r} != {self.model_dataset_name!r}"
            )
        return self

    @model_validator(mode="after")
    def label_maps_match_the_public_contract(self) -> "Settings":
        """Reject configurations that would silently relabel model outputs."""
        expected_labels = {"negative", "neutral", "positive"}
        expected_ids = set(range(self.num_labels))
        if set(self.label_map) != expected_ids:
            raise ValueError(f"label_map must contain ids {sorted(expected_ids)}")
        if set(self.label_map.values()) != expected_labels:
            raise ValueError(f"label_map must contain labels {sorted(expected_labels)}")
        inverse = {label: label_id for label_id, label in self.label_map.items()}
        if self.rev_label_map != inverse:
            raise ValueError("rev_label_map must be the inverse of label_map")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
