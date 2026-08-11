"""Deterministic splitting and the drift reference distribution.

The drift reference is built here, beside the splits, so that it always
describes the data the model was actually trained on.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftReference:
    """The training-time input distribution, logged alongside the model."""

    length_bin_edges: tuple[float, ...]
    length_bin_freqs: tuple[float, ...]
    positive_prior: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "length_bin_edges": list(self.length_bin_edges),
            "length_bin_freqs": list(self.length_bin_freqs),
            "positive_prior": self.positive_prior,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftReference":
        """Rebuild a reference from :meth:`to_dict` output."""
        return cls(
            length_bin_edges=tuple(float(x) for x in payload["length_bin_edges"]),
            length_bin_freqs=tuple(float(x) for x in payload["length_bin_freqs"]),
            positive_prior=float(payload["positive_prior"]),
        )


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Take ``n`` rows preserving the label distribution, reproducibly."""
    if n >= len(df):
        return df.reset_index(drop=True)
    fraction = n / len(df)
    rng = np.random.default_rng(seed)
    groups = df.groupby("label")
    indices: list[int] = []
    for _label, group in groups:
        k = max(1, round(len(group) * fraction))
        chosen = rng.choice(group.index.to_numpy(), size=min(k, len(group)), replace=False)
        indices.extend(chosen.tolist())
    # Shuffle, cap, and return
    rng2 = np.random.default_rng(seed)
    rng2.shuffle(indices)
    indices = indices[:n]
    return df.loc[indices].reset_index(drop=True)


def make_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve train/validation from the train frame and test from the test frame.

    Validation is drawn from the training source so the test split stays
    untouched and comparable across experiments.
    """
    pool = stratified_subsample(train_df, train_size + val_size, seed=seed)
    train = pool.iloc[:train_size].reset_index(drop=True)
    val = pool.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test = stratified_subsample(test_df, test_size, seed=seed)
    return train, val, test


def build_drift_reference(
    texts: Sequence[str], labels: Sequence[int], n_bins: int = 10
) -> DriftReference:
    """Summarise the training input distribution as quantile length bins."""
    lengths = np.array([len(t) for t in texts], dtype=float)
    edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lengths.min(), lengths.min() + 1.0])
    edges[0] = 0.0
    edges[-1] = float(lengths.max()) * 10.0

    counts, _ = np.histogram(lengths, bins=edges)
    freqs = counts / counts.sum()

    return DriftReference(
        length_bin_edges=tuple(float(e) for e in edges),
        length_bin_freqs=tuple(float(f) for f in freqs),
        positive_prior=float(np.mean(labels)),
    )
