"""Prometheus collectors and in-process drift tracking.

Drift is computed here rather than in a sidecar because it is a function of the
prediction the API just made; shipping predictions elsewhere to measure them
would add a service without adding information.
"""

from collections import deque
from collections.abc import Sequence

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Info

from sentiment.data.preprocess import DriftReference

MIN_OBSERVATIONS = 30

# ---------------------------------------------------------------------------
# HTTP metrics. Names and labels are Lab 4's, unchanged, so that lab's
# system_dashboard.json imports and works without editing a single query.
# Recorded by MetricsMiddleware, never per-endpoint.
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# ML metrics. Lab 4 names, with one deliberate change: PREDICTION_COUNT gains a
# `label` dimension, because class skew is a real signal for a classifier
# whereas Lab 4 predicted a continuous rating.
# ---------------------------------------------------------------------------
PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["label", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds",
    "Time to generate a prediction",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type", "model_version"],
)
MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Whether the ML model is loaded (1) or not (0)",
)
MODEL_INFO = Info(
    "ml_model_info",
    "Information about the loaded ML model",
)
MODEL_LAST_RELOAD = Gauge(
    "ml_model_last_reload_timestamp",
    "Unix timestamp of last model reload",
)
BATCH_SIZE = Histogram(
    "ml_batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 100],
)

# ---------------------------------------------------------------------------
# Custom metrics added by this project. Each exists because a specific alert or
# dashboard panel needs it; none is here to pad the count.
# ---------------------------------------------------------------------------
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of prediction confidence",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
LOW_CONFIDENCE = Counter(
    "ml_low_confidence_total",
    "Predictions below the low-confidence threshold",
)
INPUT_LENGTH = Histogram(
    "ml_input_length_chars",
    "Distribution of input text length in characters",
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
)
DRIFT_PSI = Gauge(
    "ml_drift_psi",
    "Population stability index of live input lengths against the training reference",
)
FAIRNESS_MAX_DELTA = Gauge(
    "ml_fairness_max_delta",
    "Maximum EEC identity-pair score delta measured for the served model",
)


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], eps: float = 1e-6
) -> float:
    """Return the PSI between two binned distributions.

    Conventional reading: below 0.1 is stable, 0.1-0.2 is a moderate shift, and
    above 0.2 is a significant shift. The alert threshold uses 0.2.

    Args:
        expected: Reference bin frequencies or counts.
        actual: Observed bin frequencies or counts, same length as ``expected``.
        eps: Floor applied to both, so empty bins do not produce infinities.

    Returns:
        The PSI as a non-negative float.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


class DriftTracker:
    """A bounded rolling window of input lengths, scored against a reference."""

    def __init__(self, reference: DriftReference, window_size: int = 1_000) -> None:
        self._edges = np.asarray(reference.length_bin_edges, dtype=float)
        self._expected = np.asarray(reference.length_bin_freqs, dtype=float)
        self._window: deque[int] = deque(maxlen=window_size)

    def __len__(self) -> int:
        """Return the number of observations currently in the window."""
        return len(self._window)

    def observe(self, text_length: int) -> None:
        """Record one observed input length."""
        self._window.append(text_length)

    def psi(self) -> float:
        """Return the current PSI, or 0.0 before the window is warm."""
        if len(self._window) < MIN_OBSERVATIONS:
            return 0.0
        values = np.clip(np.asarray(self._window, dtype=float), self._edges[0], self._edges[-1])
        counts, _ = np.histogram(values, bins=self._edges)
        return population_stability_index(self._expected, counts)
