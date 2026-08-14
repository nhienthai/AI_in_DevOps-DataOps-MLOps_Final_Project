#!/usr/bin/env python3
"""CLI script to trigger baseline or XLM-RoBERTa model training."""

# ⚠️ MUST be set before any mlflow import — works for MLflow 3.x on Kaggle
import argparse
import os
import sys

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Add the src-layout package root when the script is run without installation.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import mlflow  # noqa: E402

from sentiment.config import settings  # noqa: E402
from sentiment.training.train import train_baseline_model, train_transformer_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model (UIT-VSFC)")
    parser.add_argument(
        "--model-type",
        choices=["baseline", "transformer"],
        default="transformer",
        help="Type of model to train (baseline TF-IDF or XLM-RoBERTa transformer)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=settings.model_name,
        help="HuggingFace model identifier (e.g. xlm-roberta-base)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=settings.model_dataset_name,
        help="HuggingFace dataset name (default: tridm/UIT-VSFC)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=settings.epochs,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.batch_size,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=settings.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=settings.artifacts_dir,
        help="Output directory for trained model artifacts",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=settings.mlflow_tracking_uri,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="Optuna trials for the baseline; 0 skips the sweep",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Cross-validation folds for the baseline; 0 skips it",
    )
    parser.add_argument(
        "--mitigation",
        choices=["none", "counterfactual", "blinding"],
        default="none",
        help="Fairness mitigation applied to the baseline",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Compute and log SHAP global importance",
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("sentiment-analysis-uit-vsfc")

    print(f"=== Starting Training ({args.model_type.upper()}) ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model Name: {args.model_name}")
    print(f"MLflow URI: {args.mlflow_uri}")

    if args.model_type == "baseline":
        out_path = os.path.join(args.output_dir, "baseline_model.joblib")
        metrics = train_baseline_model(
            dataset_name=args.dataset,
            output_path=out_path,
            tune_trials=args.tune_trials,
            cv_splits=args.cv_splits,
            mitigation=args.mitigation,
            explain=args.explain,
            artifacts_dir=os.path.join(args.output_dir, "baseline"),
        )
    else:
        out_path = os.path.join(args.output_dir, "xlm-roberta")
        metrics = train_transformer_model(
            model_name=args.model_name,
            dataset_name=args.dataset,
            output_dir=out_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )

    print("=== Training Complete ===")
    print(metrics)


if __name__ == "__main__":
    main()
