#!/usr/bin/env python3
"""Quality Gate script to validate model performance and promote to MLflow Production stage."""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset
from src.sentiment.config import settings
from src.sentiment.models.baseline import BaselinePredictor
from src.sentiment.models.registry import register_and_promote_model
from src.sentiment.models.transformer import TransformerPredictor
from src.sentiment.training.evaluate import check_latency_budget, evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description="Model Quality Gate & Promotion Script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./artifacts/baseline_model.joblib",
        help="Path to trained model artifact directory or joblib file",
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "transformer"],
        default="baseline",
        help="Model architecture type",
    )
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=0.85,
        help="Minimum Macro-F1 threshold required for promotion",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        default=200.0,
        help="Maximum p95 latency allowed in ms",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional MLflow run ID to register/promote if gate passes",
    )

    args = parser.parse_args()

    print("=== Running Model Quality Gate ===")
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type}")

    if not os.path.exists(args.model_path):
        print(f"[FAIL] Model path '{args.model_path}' does not exist.")
        sys.exit(1)

    # Load predictor
    if args.model_type == "baseline":
        predictor = BaselinePredictor.load(args.model_path)
    else:
        predictor = TransformerPredictor.from_pretrained(args.model_path)

    # 1. Evaluate accuracy & macro F1 on test split
    ds = load_dataset(settings.dataset_name)
    test_texts = ds["test"]["Sentence"]
    test_labels = ds["test"]["Encoded_sentiment"]

    predictions = [p["label"] for p in predictor.predict_batch(test_texts)]
    pred_ids = [settings.rev_label_map[p] for p in predictions]

    eval_results = evaluate_predictions(test_labels, pred_ids)
    macro_f1 = eval_results["macro_f1"]
    accuracy = eval_results["accuracy"]

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f} (Required >= {args.min_macro_f1})")

    # 2. Check latency budget
    latency_results = check_latency_budget(
        predictor, p95_target_ms=args.max_p95_latency_ms
    )
    p95_ms = latency_results["p95_latency_ms"]
    print(f"p95 Latency: {p95_ms:.2f} ms (Target < {args.max_p95_latency_ms} ms)")

    # Gate decision
    passed_f1 = macro_f1 >= args.min_macro_f1
    passed_latency = p95_ms < args.max_p95_latency_ms

    if passed_f1 and passed_latency:
        print("=== [PASS] Quality Gate Passed Successfully! ===")
        if args.run_id:
            print("Promoting model to MLflow Production stage...")
            version = register_and_promote_model(run_id=args.run_id, stage="Production")
            print(f"Model version {version} promoted to Production.")
        sys.exit(0)
    else:
        print("=== [FAIL] Model Quality Gate Failed! ===")
        if not passed_f1:
            print(f" - Macro F1 ({macro_f1:.4f}) fell below minimum required ({args.min_macro_f1})")
        if not passed_latency:
            print(f" - p95 Latency ({p95_ms:.2f} ms) exceeded maximum allowed ({args.max_p95_latency_ms} ms)")
        sys.exit(1)


if __name__ == "__main__":
    main()
