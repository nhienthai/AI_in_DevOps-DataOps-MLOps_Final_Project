"""PSI and rolling drift tracker tests."""

import pytest

from sentiment.serving.metrics import DriftReference, DriftTracker, population_stability_index

REFERENCE = DriftReference(
    length_bin_edges=(0.0, 10.0, 20.0, 30.0, 1_000.0),
    length_bin_freqs=(0.25, 0.25, 0.25, 0.25),
    positive_prior=0.5,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    assert population_stability_index([0.25] * 4, [0.25] * 4) < 1e-9


def test_psi_increases_with_divergence_and_handles_empty_bins() -> None:
    mild = population_stability_index([0.25] * 4, [0.30, 0.25, 0.25, 0.20])
    severe = population_stability_index([0.25] * 4, [1.0, 0.0, 0.0, 0.0])
    assert 0 < mild < severe
    assert severe > 0.2


@pytest.mark.parametrize(
    ("expected", "actual"),
    [([], []), ([1.0], [0.5, 0.5]), ([-1.0, 2.0], [0.5, 0.5]), ([0.0], [1.0])],
)
def test_psi_rejects_invalid_distributions(expected: list[float], actual: list[float]) -> None:
    with pytest.raises(ValueError):
        population_stability_index(expected, actual)


def test_psi_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="eps"):
        population_stability_index([1.0], [1.0], eps=0)


def test_tracker_warms_up_then_detects_shift_and_bounds_its_window() -> None:
    tracker = DriftTracker(REFERENCE, window_size=50)
    for _ in range(5):
        tracker.observe(5)
    assert tracker.psi() == 0.0
    for _ in range(100):
        tracker.observe(10_000_000)
    assert len(tracker) == 50
    assert tracker.psi() > 0.2


def test_tracker_reports_low_psi_for_matching_traffic() -> None:
    tracker = DriftTracker(REFERENCE, window_size=1_000)
    for length in [5, 15, 25, 100] * 50:
        tracker.observe(length)
    assert tracker.psi() < 0.1


def test_tracker_rejects_bad_reference_window_and_observation() -> None:
    with pytest.raises(ValueError, match="frequencies"):
        DriftTracker(DriftReference((0.0, 1.0), (0.5, 0.5), 0.5), window_size=30)
    with pytest.raises(ValueError, match="two bin"):
        DriftTracker(DriftReference((0.0,), (), 0.5), window_size=30)
    with pytest.raises(ValueError, match="window_size"):
        DriftTracker(REFERENCE, window_size=10)
    tracker = DriftTracker(REFERENCE)
    with pytest.raises(ValueError, match="negative"):
        tracker.observe(-1)
