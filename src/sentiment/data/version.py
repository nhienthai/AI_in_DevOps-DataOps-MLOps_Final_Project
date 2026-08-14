"""Content-addressed dataset versioning.

A model is only reproducible if you can say which data produced it. Rather than
adding a DVC remote for a dataset that is already immutable on the HuggingFace
hub, this module fingerprints exactly what the training run consumed: a stable
hash over the text and labels, the row counts, and the class distribution.

The fingerprint goes into MLflow as params, so two runs that disagree on metrics
can be checked for whether they even saw the same data. A hash mismatch between
a run and the current dataset means the data moved underneath the experiment.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import log2
from pathlib import Path
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class DatasetVersion:
    """Reproducibility fingerprint for one dataset split."""

    name: str
    split: str
    n_rows: int
    content_sha256: str
    label_distribution: dict[str, int]
    label_entropy: float

    def as_params(self) -> dict[str, str]:
        """Return a flat mapping suitable for ``mlflow.log_params``."""
        return {
            f"data.{self.split}.name": self.name,
            f"data.{self.split}.rows": str(self.n_rows),
            f"data.{self.split}.sha256": self.content_sha256,
            f"data.{self.split}.labels": json.dumps(self.label_distribution, sort_keys=True),
            f"data.{self.split}.entropy": f"{self.label_entropy:.6f}",
        }


def _label_entropy(counts: Mapping[str, int]) -> float:
    """Return the Shannon entropy of a label distribution, in bits.

    Zero means every row carries the same label. For three balanced classes the
    maximum is log2(3), about 1.585, so a value far below that is the signature
    of the class imbalance this dataset has.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        share = count / total
        entropy -= share * log2(share)
    return entropy


def fingerprint(
    name: str, split: str, texts: Sequence[str], labels: Sequence[int]
) -> DatasetVersion:
    """Fingerprint one split by hashing its content in order.

    Args:
        name: Dataset identifier, for example ``tridm/UIT-VSFC``.
        split: Split name, for example ``train``.
        texts: The text column, in the order the trainer will see it.
        labels: The integer label column, aligned with ``texts``.

    Returns:
        A :class:`DatasetVersion` describing the split.

    Raises:
        ValueError: If ``texts`` and ``labels`` differ in length.
    """
    if len(texts) != len(labels):
        raise ValueError(f"texts and labels differ: {len(texts)} vs {len(labels)}")

    digest = hashlib.sha256()
    for text, label in zip(texts, labels):
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(text.encode("utf-8"))
        digest.update(b"\x1e")

    counts = {str(key): value for key, value in sorted(Counter(labels).items())}
    return DatasetVersion(
        name=name,
        split=split,
        n_rows=len(texts),
        content_sha256=digest.hexdigest(),
        label_distribution=counts,
        label_entropy=_label_entropy(counts),
    )


def fingerprint_all(
    name: str, splits: Mapping[str, tuple[Sequence[str], Sequence[int]]]
) -> dict[str, DatasetVersion]:
    """Fingerprint several splits at once."""
    return {
        split: fingerprint(name, split, texts, labels) for split, (texts, labels) in splits.items()
    }


def write_manifest(versions: Iterable[DatasetVersion], path: Path) -> Path:
    """Write a JSON manifest of dataset versions, for logging as an artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {version.split: asdict(version) for version in versions}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def combined_params(versions: Iterable[DatasetVersion]) -> dict[str, str]:
    """Flatten several dataset versions into one MLflow params mapping."""
    params: dict[str, str] = {}
    for version in versions:
        params.update(version.as_params())
    return params
