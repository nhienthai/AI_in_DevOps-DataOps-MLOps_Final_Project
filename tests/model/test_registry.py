"""MLflow registry artifact discovery tests without a live registry."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentiment.models.registry import (
    _find_transformer_directory,
    _latest_version,
    _load_drift_reference,
)


def test_find_transformer_directory_handles_nested_artifact(tmp_path: Path) -> None:
    model = tmp_path / "checkpoint" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    assert _find_transformer_directory(tmp_path) == model


def test_find_transformer_directory_rejects_bad_artifact(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="config.json"):
        _find_transformer_directory(tmp_path)


def test_load_drift_reference(tmp_path: Path) -> None:
    payload = {
        "length_bin_edges": [0, 10, 100],
        "length_bin_freqs": [0.4, 0.6],
        "positive_prior": 0.55,
    }
    (tmp_path / "drift_reference.json").write_text(json.dumps(payload), encoding="utf-8")
    reference = _load_drift_reference(tmp_path)
    assert reference is not None
    assert reference.length_bin_freqs == (0.4, 0.6)


def test_latest_version_selects_highest_version() -> None:
    client = SimpleNamespace(
        get_latest_versions=lambda _name, stages: [
            SimpleNamespace(version="2"),
            SimpleNamespace(version="10"),
        ]
    )
    assert _latest_version(client, "sentiment", "Production").version == "10"
