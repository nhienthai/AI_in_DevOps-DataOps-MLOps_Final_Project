"""Dataset fingerprinting: reproducibility depends on these being exact."""

import json
from pathlib import Path

import pytest

from sentiment.data.version import combined_params, fingerprint, fingerprint_all, write_manifest

TEXTS = ["thầy dạy hay", "bài giảng khó hiểu", "phòng học bình thường"]
LABELS = [2, 0, 1]


def test_same_content_yields_the_same_hash() -> None:
    first = fingerprint("ds", "train", TEXTS, LABELS)
    second = fingerprint("ds", "train", list(TEXTS), list(LABELS))
    assert first.content_sha256 == second.content_sha256


def test_changed_label_changes_the_hash() -> None:
    original = fingerprint("ds", "train", TEXTS, LABELS)
    altered = fingerprint("ds", "train", TEXTS, [2, 0, 2])
    assert original.content_sha256 != altered.content_sha256


def test_changed_text_changes_the_hash() -> None:
    original = fingerprint("ds", "train", TEXTS, LABELS)
    altered = fingerprint("ds", "train", ["thầy dạy hay!", *TEXTS[1:]], LABELS)
    assert original.content_sha256 != altered.content_sha256


def test_reordering_changes_the_hash() -> None:
    """Order matters because the trainer sees the rows in order."""
    original = fingerprint("ds", "train", TEXTS, LABELS)
    reordered = fingerprint("ds", "train", TEXTS[::-1], LABELS[::-1])
    assert original.content_sha256 != reordered.content_sha256


def test_delimiters_prevent_field_collisions() -> None:
    """Concatenating without separators would make these two datasets identical."""
    first = fingerprint("ds", "train", ["ab", "c"], [0, 0])
    second = fingerprint("ds", "train", ["a", "bc"], [0, 0])
    assert first.content_sha256 != second.content_sha256


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="differ"):
        fingerprint("ds", "train", TEXTS, [1, 2])


def test_distribution_and_entropy() -> None:
    version = fingerprint("ds", "train", TEXTS, LABELS)
    assert version.n_rows == 3
    assert version.label_distribution == {"0": 1, "1": 1, "2": 1}
    assert version.label_entropy == pytest.approx(1.58496, abs=1e-4)


def test_single_class_has_zero_entropy() -> None:
    version = fingerprint("ds", "train", TEXTS, [1, 1, 1])
    assert version.label_entropy == pytest.approx(0.0)


def test_params_are_flat_strings_for_mlflow() -> None:
    params = fingerprint("ds", "train", TEXTS, LABELS).as_params()
    assert params["data.train.rows"] == "3"
    assert params["data.train.name"] == "ds"
    assert json.loads(params["data.train.labels"]) == {"0": 1, "1": 1, "2": 1}
    assert all(isinstance(value, str) for value in params.values())


def test_combined_params_span_every_split() -> None:
    versions = fingerprint_all("ds", {"train": (TEXTS, LABELS), "test": (TEXTS[:2], LABELS[:2])})
    params = combined_params(versions.values())
    assert params["data.train.rows"] == "3"
    assert params["data.test.rows"] == "2"


def test_manifest_round_trips(tmp_path: Path) -> None:
    versions = fingerprint_all("ds", {"train": (TEXTS, LABELS)})
    path = write_manifest(versions.values(), tmp_path / "nested" / "manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["train"]["n_rows"] == 3
    assert payload["train"]["content_sha256"]
