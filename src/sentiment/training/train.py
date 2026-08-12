"""Training pipeline for baseline and XLM-RoBERTa transformer models with MLflow logging."""

import logging
import os
from typing import Any, Dict
# ⚠️ MUST be set before mlflow import for MLflow 3.x compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sentiment.config import settings  # noqa: E402
from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.training.evaluate import compute_metrics, evaluate_predictions  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_baseline_model(
    dataset_name: str = settings.dataset_name,
    output_path: str = "./artifacts/baseline_model.joblib",
) -> Dict[str, Any]:
    """Train TF-IDF + LogisticRegression baseline model and log to MLflow."""
    logger.info("Loading dataset '%s' for baseline training...", dataset_name)
    ds = load_dataset(dataset_name)

    train_texts = ds["train"]["Sentence"]
    train_labels = ds["train"]["Encoded_sentiment"]
    test_texts = ds["test"]["Sentence"]
    test_labels = ds["test"]["Encoded_sentiment"]

    with mlflow.start_run(run_name="baseline-tfidf-logreg"):
        mlflow.log_param("model_type", "baseline_tfidf_logreg")
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("num_train_samples", len(train_texts))

        predictor = BaselinePredictor()
        predictor.fit(train_texts, train_labels)

        # Evaluate
        preds = [p.label for p in predictor.predict(test_texts)]
        pred_ids = [settings.rev_label_map[p] for p in preds]

        metrics = evaluate_predictions(test_labels, pred_ids)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)

        logger.info("Baseline Test Macro F1: %.4f", metrics["macro_f1"])

        predictor.save(output_path)
        mlflow.log_artifact(output_path, artifact_path="model")

        return metrics


def train_transformer_model(
    model_name: str = settings.model_name,
    dataset_name: str = settings.dataset_name,
    output_dir: str = "./artifacts/xlm-roberta",
    epochs: int = settings.epochs,
    batch_size: int = settings.batch_size,
    learning_rate: float = settings.learning_rate,
    max_length: int = settings.max_length,
) -> Dict[str, Any]:
    """Fine-tune XLM-RoBERTa on UIT-VSFC and log experiments to MLflow."""
    logger.info("Loading dataset '%s'...", dataset_name)
    ds = load_dataset(dataset_name)

    logger.info("Loading tokenizer and model for '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(
            examples["Sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    tokenized_ds = tokenized_ds.rename_column("Encoded_sentiment", "label")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=settings.num_labels
    )

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=use_fp16,
        report_to=[],  # We manage MLflow manually via mlflow.start_run()
        dataloader_num_workers=2,
    )

    with mlflow.start_run(run_name=f"fine-tune-{model_name.replace('/', '-')}"):
        mlflow.log_params({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
        })

        val_split = "validation" if "validation" in tokenized_ds else "dev"
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds[val_split],
            compute_metrics=compute_metrics,
        )

        logger.info("Starting Transformer fine-tuning...")
        trainer.train()

        # Final evaluation on test split
        test_results = trainer.evaluate(tokenized_ds["test"])
        for k, v in test_results.items():
            clean_key = k.replace("eval_", "test_")
            mlflow.log_metric(clean_key, v)

        logger.info("Test Results: %s", test_results)

        # Save model & tokenizer
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        mlflow.log_artifacts(output_dir, artifact_path="model")

        return test_results
