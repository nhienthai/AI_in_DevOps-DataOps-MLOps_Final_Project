"""Evaluation metrics and latency benchmarking for sentiment analysis models."""

import time
from typing import Any, Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from src.sentiment.config import settings


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def evaluate_predictions(y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
    """Compute detailed evaluation report dictionary."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=[settings.label_map[i] for i in sorted(settings.label_map.keys())],
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
    }


def check_latency_budget(
    predictor: Any,
    sample_text: str = "Bài giảng rất dễ hiểu và giảng viên nhiệt tình.",
    p95_target_ms: float = 200.0,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark prediction latency and verify against p95 SLA target."""
    latencies = []
    # Warmup run
    _ = predictor.predict(sample_text)

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = predictor.predict(sample_text)
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies.append(duration_ms)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    p99_ms = float(np.percentile(latencies, 99))
    passed_sla = bool(p95_ms < p95_target_ms)

    return {
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "p99_latency_ms": round(p99_ms, 2),
        "p95_target_ms": p95_target_ms,
        "passed_sla": passed_sla,
    }
