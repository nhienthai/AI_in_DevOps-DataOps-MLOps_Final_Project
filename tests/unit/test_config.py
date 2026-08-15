"""Configuration contract tests."""

import pytest
from pydantic import ValidationError

from sentiment.config import Settings, get_settings


def test_defaults_match_the_service_contract() -> None:
    settings = Settings()
    assert settings.dataset_name == "tridm/UIT-VSFC"
    assert settings.dataset_name == settings.model_dataset_name
    assert settings.train_size == 200_000
    assert settings.val_size == 25_000
    assert settings.test_size == 25_000
    assert settings.random_seed == 42
    assert settings.max_batch_size == 64
    assert settings.max_text_length == 5_000
    assert settings.low_confidence_threshold == 0.7
    assert settings.drift_window_size == 1_000
    assert settings.predictor_backend == "stub"
    assert settings.model_dataset_name == "tridm/UIT-VSFC"


def test_sentiment_environment_prefix_overrides_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTIMENT_MAX_BATCH_SIZE", "8")
    assert Settings().max_batch_size == 8


def test_invalid_batch_size_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(max_batch_size=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_labels": 2},
        {"label_map": {0: "negative", 1: "positive", 2: "positive"}},
        {"rev_label_map": {"negative": 0, "neutral": 2, "positive": 1}},
    ],
)
def test_inconsistent_label_contract_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(**overrides)


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_blank_reload_token_disables_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_RELOAD_TOKEN", "")
    get_settings.cache_clear()
    try:
        assert get_settings().reload_token is None
    finally:
        get_settings.cache_clear()
