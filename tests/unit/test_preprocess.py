import pandas as pd

from sentiment.data.preprocess import (
    DriftReference,
    build_drift_reference,
    make_splits,
    stratified_subsample,
)


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"label": [i % 2 for i in range(n)], "text": [f"review {i}" for i in range(n)]}
    )


def test_subsample_returns_requested_size():
    assert len(stratified_subsample(frame(1000), 100, seed=42)) == 100


def test_subsample_preserves_label_balance():
    out = stratified_subsample(frame(1000), 100, seed=42)
    assert abs(out["label"].mean() - 0.5) < 0.05


def test_subsample_is_deterministic_for_a_seed():
    a = stratified_subsample(frame(1000), 100, seed=42)
    b = stratified_subsample(frame(1000), 100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_subsample_larger_than_source_returns_all():
    assert len(stratified_subsample(frame(50), 500, seed=42)) == 50


def test_make_splits_sizes_and_disjointness():
    train, val, test = make_splits(
        frame(1000), frame(400), train_size=200, val_size=50, test_size=40, seed=42
    )
    assert (len(train), len(val), len(test)) == (200, 50, 40)
    assert set(train["text"]).isdisjoint(set(val["text"]))


def test_drift_reference_frequencies_sum_to_one():
    ref = build_drift_reference(
        [("x" * (i % 50 + 1)) for i in range(500)], [i % 2 for i in range(500)]
    )
    assert abs(sum(ref.length_bin_freqs) - 1.0) < 1e-9
    assert len(ref.length_bin_freqs) == len(ref.length_bin_edges) - 1


def test_drift_reference_records_positive_prior():
    ref = build_drift_reference(["a", "bb", "ccc", "dddd"], [1, 1, 0, 0])
    assert ref.positive_prior == 0.5


def test_drift_reference_roundtrips_through_dict():
    ref = build_drift_reference(
        [("x" * (i % 30 + 1)) for i in range(200)], [i % 2 for i in range(200)]
    )
    assert DriftReference.from_dict(ref.to_dict()) == ref
