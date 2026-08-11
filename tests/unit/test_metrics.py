from sentiment.data.preprocess import DriftReference
from sentiment.serving.metrics import DriftTracker, population_stability_index

REF = DriftReference(
    length_bin_edges=(0.0, 10.0, 20.0, 30.0, 1000.0),
    length_bin_freqs=(0.25, 0.25, 0.25, 0.25),
    positive_prior=0.5,
)


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index([0.25] * 4, [0.25] * 4) < 1e-9


def test_psi_grows_with_divergence():
    mild = population_stability_index([0.25] * 4, [0.30, 0.25, 0.25, 0.20])
    severe = population_stability_index([0.25] * 4, [0.85, 0.05, 0.05, 0.05])
    assert 0 < mild < severe


def test_psi_flags_significant_shift_above_conventional_threshold():
    assert population_stability_index([0.25] * 4, [0.85, 0.05, 0.05, 0.05]) > 0.2


def test_psi_tolerates_empty_bins():
    assert population_stability_index([0.25] * 4, [1.0, 0.0, 0.0, 0.0]) > 0.0


def test_tracker_reports_zero_before_enough_observations():
    tracker = DriftTracker(REF, window_size=100)
    for _ in range(5):
        tracker.observe(5)
    assert tracker.psi() == 0.0


def test_tracker_reports_low_psi_on_matching_traffic():
    tracker = DriftTracker(REF, window_size=1000)
    for length in [5, 15, 25, 100] * 50:
        tracker.observe(length)
    assert tracker.psi() < 0.1


def test_tracker_reports_high_psi_on_shifted_traffic():
    tracker = DriftTracker(REF, window_size=1000)
    for _ in range(200):
        tracker.observe(5)
    assert tracker.psi() > 0.2


def test_tracker_window_is_bounded():
    tracker = DriftTracker(REF, window_size=50)
    for i in range(500):
        tracker.observe(i)
    assert len(tracker) == 50


def test_tracker_clips_lengths_beyond_the_reference_range():
    tracker = DriftTracker(REF, window_size=100)
    for _ in range(50):
        tracker.observe(10_000_000)
    assert tracker.psi() > 0.0
