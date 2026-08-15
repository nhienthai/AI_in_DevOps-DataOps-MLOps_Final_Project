"""Evaluation metrics and latency benchmarking for sentiment analysis models."""

import time
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from sentiment.config import settings


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
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


def cross_validate_baseline(
    texts: List[str],
    labels: List[int],
    n_splits: int = 5,
    seed: int = 42,
    tfidf_params: Dict[str, Any] | None = None,
    clf_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run stratified k-fold cross-validation on the baseline pipeline.

    A single held-out split cannot distinguish a genuinely better model from a
    luckier split, which matters here because ``neutral`` is only about 4% of
    UIT-VSFC — one fold can differ from another by several F1 points on that
    class alone. Reporting the standard deviation alongside the mean is what
    makes a comparison between two models honest.

    Stratification keeps each fold's class proportions equal to the whole, so no
    fold ends up with too few ``neutral`` rows to score.

    Args:
        texts: Training texts.
        labels: Integer labels aligned with ``texts``.
        n_splits: Number of folds.
        seed: Seed for the fold shuffle, so folds are reproducible.
        tfidf_params: Overrides for :class:`TfidfVectorizer`.
        clf_params: Overrides for :class:`LogisticRegression`.

    Returns:
        Per-fold scores plus the mean and standard deviation of each metric.

    Raises:
        ValueError: If ``texts`` and ``labels`` differ in length.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline

    if len(texts) != len(labels):
        raise ValueError(f"texts and labels differ: {len(texts)} vs {len(labels)}")

    vectoriser_kwargs: Dict[str, Any] = {"ngram_range": (1, 2), "max_features": 10000}
    vectoriser_kwargs.update(tfidf_params or {})
    classifier_kwargs: Dict[str, Any] = {
        "max_iter": 1000,
        "C": 1.0,
        "random_state": seed,
        "class_weight": "balanced",
    }
    classifier_kwargs.update(clf_params or {})

    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    text_array = np.asarray(texts, dtype=object)
    label_array = np.asarray(labels)

    per_fold: List[Dict[str, float]] = []
    for train_index, test_index in folds.split(text_array, label_array):
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectoriser_kwargs)),
                ("clf", LogisticRegression(**classifier_kwargs)),
            ]
        )
        pipeline.fit(text_array[train_index].tolist(), label_array[train_index].tolist())
        predictions = pipeline.predict(text_array[test_index].tolist())
        scores = evaluate_predictions(
            label_array[test_index].tolist(), [int(value) for value in predictions]
        )
        per_fold.append(
            {
                "accuracy": scores["accuracy"],
                "macro_f1": scores["macro_f1"],
                "macro_precision": scores["macro_precision"],
                "macro_recall": scores["macro_recall"],
            }
        )

    summary: Dict[str, Any] = {"n_splits": n_splits, "seed": seed, "folds": per_fold}
    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        values = [fold[metric] for fold in per_fold]
        summary[f"cv_{metric}_mean"] = float(np.mean(values))
        summary[f"cv_{metric}_std"] = float(np.std(values))
    return summary


def check_latency_budget(
    predictor: Any,
    sample_text: str = "Bài giảng rất dễ hiểu và giảng viên nhiệt tình.",
    p95_target_ms: float = 200.0,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Benchmark prediction latency and verify against p95 SLA target."""
    latencies = []
    # Warmup run
    _ = predictor.predict([sample_text])

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = predictor.predict([sample_text])
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
