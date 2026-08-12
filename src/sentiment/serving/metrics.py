"""Prometheus collectors and in-process input-drift tracking."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, Info

MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class DriftReference:
    """Dependency-light serving view of the training drift artifact."""

    length_bin_edges: tuple[float, ...]
    length_bin_freqs: tuple[float, ...]
    positive_prior: float


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
MODEL_LOADED = Gauge("ml_model_loaded", "Whether the ML model is loaded")
MODEL_INFO = Info("ml_model_info", "Information about the loaded ML model")
MODEL_LAST_RELOAD = Gauge(
    "ml_model_last_reload_timestamp", "Unix timestamp of the last model reload"
)
MODEL_LOAD_DURATION = Histogram(
    "ml_model_load_duration_seconds", "Time required to load and warm a model"
)
MODEL_LOAD_FAILURES = Counter("ml_model_load_failures_total", "Model load and warm-up failures")
BATCH_SIZE = Histogram(
    "ml_batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 64],
)
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of prediction confidence",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
LOW_CONFIDENCE = Counter(
    "ml_low_confidence_total", "Predictions below the configured confidence threshold"
)
INPUT_LENGTH = Histogram(
    "ml_input_length_chars",
    "Distribution of input length in characters",
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
)
DRIFT_PSI = Gauge("ml_drift_psi", "Population stability index against the training reference")
FAIRNESS_MAX_DELTA = Gauge("ml_fairness_max_delta", "Maximum EEC identity-pair score delta")
INFERENCE_IN_PROGRESS = Gauge("ml_inference_in_progress", "Inference batches currently executing")
INFERENCE_QUEUE_DEPTH = Gauge(
    "ml_inference_queue_depth", "Requests waiting for an inference worker"
)
INFERENCE_QUEUE_LATENCY = Histogram(
    "ml_inference_queue_duration_seconds", "Time spent waiting for an inference worker"
)
INFERENCE_TIMEOUTS = Counter(
    "ml_inference_timeouts_total", "Inference batches exceeding the configured timeout"
)
INFERENCE_OVERLOADS = Counter(
    "ml_inference_overloads_total", "Requests rejected because inference capacity is full"
)
INPUT_TRUNCATIONS = Counter("ml_input_truncations_total", "Inputs truncated by the active model")


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], eps: float = 1e-6
) -> float:
    """Return PSI for two equal-length non-negative binned distributions."""
    if len(expected) != len(actual) or len(expected) == 0:
        raise ValueError("expected and actual must be non-empty and have equal lengths")
    if eps <= 0:
        raise ValueError("eps must be positive")

    expected_array: Any = np.asarray(expected, dtype=float)
    actual_array: Any = np.asarray(actual, dtype=float)
    if np.any(expected_array < 0) or np.any(actual_array < 0):
        raise ValueError("distributions cannot contain negative values")
    if expected_array.sum() == 0 or actual_array.sum() == 0:
        raise ValueError("distributions must contain at least one positive value")

    expected_array = np.clip(expected_array, eps, None)
    actual_array = np.clip(actual_array, eps, None)
    expected_array /= expected_array.sum()
    actual_array /= actual_array.sum()
    return float(np.sum((actual_array - expected_array) * np.log(actual_array / expected_array)))


class DriftTracker:
    """Maintain a bounded rolling window of observed input lengths."""

    def __init__(self, reference: DriftReference, window_size: int = 1_000) -> None:
        if len(reference.length_bin_edges) < 2:
            raise ValueError("drift reference needs at least two bin edges")
        if len(reference.length_bin_freqs) != len(reference.length_bin_edges) - 1:
            raise ValueError("drift frequencies must match the number of bins")
        if window_size < MIN_OBSERVATIONS:
            raise ValueError(f"window_size must be at least {MIN_OBSERVATIONS}")
        self._edges: Any = np.asarray(reference.length_bin_edges, dtype=float)
        self._expected: Any = np.asarray(reference.length_bin_freqs, dtype=float)
        self._window: deque[int] = deque(maxlen=window_size)

    def __len__(self) -> int:
        """Return the current observation count."""
        return len(self._window)

    def observe(self, text_length: int) -> None:
        """Record one non-negative input length."""
        if text_length < 0:
            raise ValueError("text_length cannot be negative")
        self._window.append(text_length)

    def psi(self) -> float:
        """Return current PSI, or zero while the window is warming up."""
        if len(self._window) < MIN_OBSERVATIONS:
            return 0.0
        values: Any = np.clip(
            np.asarray(self._window, dtype=float), self._edges[0], self._edges[-1]
        )
        histogram: Any = np.histogram(values, bins=self._edges)
        counts: Any = histogram[0]
        return population_stability_index(self._expected.tolist(), counts.tolist())
