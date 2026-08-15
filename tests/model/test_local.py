"""Local-directory predictor loading without pulling 1 GB of weights."""

import json
from pathlib import Path
from typing import Any

import pytest

from sentiment.config import Settings
from sentiment.models.local import _read_serving_metadata, load_local_predictor


class _FakePredictor:
    """Stands in for TransformerPredictor so tests stay off the real weights."""

    def __init__(self, directory: str, model_version: str, input_format: Any = None) -> None:
        self.directory = directory
        self.version = model_version
        self.input_format = input_format
        self.stage = "Development"
        self.metrics: dict[str, float] = {}
        self.fairness_delta: float | None = None
        self.trained_at: str | None = None
        self.run_id: str | None = None

    def predict(self, texts: Any) -> list:
        return []


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "xlm-roberta"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"")
    return directory


def _settings(model_dir: Path) -> Settings:
    return Settings(predictor_backend="local", local_model_dir=model_dir)


def _patch_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sentiment.models.transformer.TransformerPredictor.from_pretrained",
        lambda save_directory, model_version="x", device=None, input_format=None: _FakePredictor(
            save_directory, model_version, input_format
        ),
    )


def test_missing_directory_names_the_setup_script(tmp_path: Path) -> None:
    settings = Settings(predictor_backend="local", local_model_dir=tmp_path / "absent")
    with pytest.raises(RuntimeError, match="setup_local_model.py"):
        load_local_predictor(settings)


def test_directory_without_config_is_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    settings = Settings(predictor_backend="local", local_model_dir=empty)
    with pytest.raises(RuntimeError, match="config.json"):
        load_local_predictor(settings)


def test_metadata_populates_serving_fields(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (model_dir / "serving_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "18476d2792b046159ad037a03899569f",
                "run_name": "fine-tune-xlm-roberta-base",
                "start_time_ms": 1786551079295,
                "metrics": {"test_macro_f1": 0.8337, "fairness_max_delta": 0.02},
            }
        ),
        encoding="utf-8",
    )
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.run_id == "18476d2792b046159ad037a03899569f"
    assert predictor.metrics["test_macro_f1"] == pytest.approx(0.8337)
    assert predictor.fairness_delta == pytest.approx(0.02)
    assert predictor.stage == "Production"
    assert predictor.trained_at is not None


def test_version_falls_back_to_directory_name(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory dropped in by hand still loads, just without lineage."""
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.version == "local-xlm-roberta"
    assert predictor.run_id is None
    assert predictor.metrics == {}


def test_version_prefers_the_donated_run(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (model_dir / "serving_metadata.json").write_text(
        json.dumps({"run_id": "abcdef1234567890", "metrics": {}}), encoding="utf-8"
    )
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.version == "local-abcdef12"


def test_unreadable_metadata_does_not_block_serving(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupt lineage costs the dashboard some labels, not the whole service."""
    (model_dir / "serving_metadata.json").write_text("{not json", encoding="utf-8")
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.metrics == {}


def test_serving_factory_routes_local_backend(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HTTP layer must reach the local loader, not fall through to MLflow."""
    from sentiment.serving.app import _default_predictor_factory

    _patch_transformer(monkeypatch)
    monkeypatch.setattr(
        "sentiment.models.registry.load_production_predictor",
        lambda **_kwargs: pytest.fail("local backend must not touch the registry"),
    )

    predictor = _default_predictor_factory(_settings(model_dir))

    assert isinstance(predictor, _FakePredictor)
    assert predictor.directory == str(model_dir)


def test_preprocessing_metadata_reaches_the_predictor(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A topic-conditioned model must be served the shape it was trained on."""
    (model_dir / "serving_metadata.json").write_text(
        json.dumps(
            {
                "metrics": {},
                "preprocessing": {
                    "clean_dataset_artifacts": True,
                    "template": "Chủ đề: others | {text}",
                },
            }
        ),
        encoding="utf-8",
    )
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.input_format.apply("lúc 9doubledot30") == "Chủ đề: others | lúc 9:30"


def test_model_without_preprocessing_gets_identity_format(
    model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_transformer(monkeypatch)

    predictor: Any = load_local_predictor(_settings(model_dir))

    assert predictor.input_format.is_identity


def test_read_serving_metadata_ignores_non_numeric_metrics(model_dir: Path) -> None:
    (model_dir / "serving_metadata.json").write_text(
        json.dumps({"metrics": {"test_macro_f1": 0.83, "note": "best run"}}), encoding="utf-8"
    )
    metadata = _read_serving_metadata(model_dir)
    assert metadata["metrics"] == {"test_macro_f1": 0.83}
