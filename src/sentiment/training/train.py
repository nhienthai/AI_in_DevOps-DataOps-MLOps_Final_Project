"""Training pipeline for baseline and XLM-RoBERTa transformer models with MLflow logging."""

import json
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# ⚠️ MUST be set before mlflow import for MLflow 3.x compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import logging
import re
from typing import Any, Dict, Optional
import mlflow
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from sentiment.config import settings  # noqa: E402
from sentiment.data.preprocess import build_drift_reference  # noqa: E402
from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.training.evaluate import (  # noqa: E402
    compute_metrics,
    cross_validate_baseline,
    evaluate_predictions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_validated_splits(dataset_name: str) -> Dict[str, Any]:
    """Load a dataset, normalise it, and put every split through the quality gate.

    This is the join that W2-04 was about. Training used to call
    ``load_dataset`` directly, so the gate in ``data/validate.py`` guarded data
    that no model was ever trained on. Everything now enters through one path, and
    a bad split raises instead of quietly degrading the model.

    Returns:
        Mapping from split name to ``{"texts", "labels", "report", "version"}``.

    Raises:
        DataQualityError: If any split fails the gate.
    """
    from sentiment.data.ingest import normalise_for
    from sentiment.data.validate import validate
    from sentiment.data.version import fingerprint

    raw = load_dataset(dataset_name)
    splits: Dict[str, Any] = {}
    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            continue
        frame = normalise_for(dataset_name, raw[split_name].to_pandas())
        report = validate(frame, min_rows=500)
        texts = frame["text"].astype(str).tolist()
        labels = [int(value) for value in frame["label"].tolist()]
        splits[split_name] = {
            "texts": texts,
            "labels": labels,
            "report": report,
            "version": fingerprint(dataset_name, split_name, texts, labels),
        }
        logger.info(
            "%s: %d rows, classes %s, rarest share %.4f, sha256 %s",
            split_name,
            report.n_rows,
            sorted(report.class_shares),
            report.min_class_share,
            splits[split_name]["version"].content_sha256[:12],
        )
    if "train" not in splits or "test" not in splits:
        raise ValueError(f"dataset '{dataset_name}' must provide train and test splits")
    return splits


def train_baseline_model(
    dataset_name: str = settings.dataset_name,
    output_path: str = "./artifacts/baseline_model.joblib",
    tune_trials: int = 0,
    cv_splits: int = 5,
    mitigation: str = "none",
    explain: bool = False,
    artifacts_dir: str = "./artifacts/baseline",
) -> Dict[str, Any]:
    """Train the TF-IDF + LogisticRegression baseline and log everything to MLflow.

    Args:
        dataset_name: HuggingFace dataset identifier.
        output_path: Where the fitted pipeline is written.
        tune_trials: Optuna trials to run. Zero skips the search and uses defaults.
        cv_splits: Folds for cross-validation. Zero skips it.
        mitigation: Fairness mitigation to apply. ``"none"``, ``"counterfactual"``
            (identity-swapped training copies) or ``"blinding"`` (identity terms
            replaced by a neutral placeholder at fit and predict time).
        explain: Compute and log SHAP global importance.
        artifacts_dir: Directory for reports and charts logged to MLflow.

    Returns:
        Held-out test metrics, plus cross-validation and tuning summaries.

    Raises:
        ValueError: If ``mitigation`` is not a known strategy.
    """
    if mitigation not in {"none", "counterfactual", "blinding"}:
        raise ValueError(f"unknown mitigation strategy: {mitigation!r}")
    from sentiment.data.version import combined_params, write_manifest

    logger.info("Loading dataset '%s' for baseline training...", dataset_name)
    splits = load_validated_splits(dataset_name)

    train_texts = splits["train"]["texts"]
    train_labels = splits["train"]["labels"]
    test_texts = splits["test"]["texts"]
    test_labels = splits["test"]["labels"]

    suffix = "" if mitigation == "none" else f"-{mitigation}"
    run_name = f"baseline-tfidf-logreg{suffix}"
    output_root = Path(artifacts_dir)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_param("model_type", "baseline_tfidf_logreg")
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("mitigation", mitigation)
        mlflow.log_param("num_train_samples", len(train_texts))

        versions = [splits[name]["version"] for name in splits]
        mlflow.log_params(combined_params(versions))
        manifest = write_manifest(versions, output_root / "dataset_manifest.json")
        mlflow.log_artifact(str(manifest), artifact_path="data")

        for name, payload in splits.items():
            report = payload["report"]
            mlflow.log_metric(f"data_{name}_rows", report.n_rows)
            mlflow.log_metric(f"data_{name}_min_class_share", report.min_class_share)

        if mitigation == "counterfactual":
            from sentiment.responsible.fairness import counterfactual_augment

            before_rows = len(train_texts)
            train_texts, train_labels = counterfactual_augment(train_texts, train_labels)
            mlflow.log_metric("augmented_rows", len(train_texts) - before_rows)
            logger.info("counterfactual augmentation: %d -> %d rows", before_rows, len(train_texts))

        tfidf_params: Dict[str, Any] | None = None
        clf_params: Dict[str, Any] | None = None
        tuning: Dict[str, Any] | None = None
        if tune_trials > 0:
            from sentiment.training.tune import tune_baseline

            tuning = tune_baseline(train_texts, train_labels, n_trials=tune_trials, n_splits=3)
            tfidf_params = tuning["best_tfidf_params"]
            clf_params = tuning["best_clf_params"]

        cv_summary: Dict[str, Any] | None = None
        if cv_splits > 0:
            cv_summary = cross_validate_baseline(
                train_texts,
                train_labels,
                n_splits=cv_splits,
                tfidf_params=tfidf_params,
                clf_params=clf_params,
            )
            mlflow.log_metrics(
                {
                    key: value
                    for key, value in cv_summary.items()
                    if key.startswith("cv_") and isinstance(value, float)
                }
            )
            logger.info(
                "cross-validated macro-F1 %.4f +/- %.4f",
                cv_summary["cv_macro_f1_mean"],
                cv_summary["cv_macro_f1_std"],
            )

        predictor = BaselinePredictor(blind_identity=mitigation == "blinding")
        predictor.fit(train_texts, train_labels, tfidf_params=tfidf_params, clf_params=clf_params)

        pred_ids = [
            settings.rev_label_map[prediction.label] for prediction in predictor.predict(test_texts)
        ]
        metrics = evaluate_predictions(test_labels, pred_ids)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"test_{key}", value)

        logger.info(
            "held-out test macro-F1 %.4f, accuracy %.4f", metrics["macro_f1"], metrics["accuracy"]
        )

        drift_reference = build_drift_reference(test_texts, test_labels)
        drift_path = output_root / "drift_reference.json"
        drift_path.parent.mkdir(parents=True, exist_ok=True)
        drift_path.write_text(json.dumps(drift_reference.to_dict()), encoding="utf-8")

        predictor.save(output_path)
        mlflow.log_artifact(output_path, artifact_path="model")
        mlflow.log_artifact(str(drift_path), artifact_path="model")

        if explain:
            from sentiment.responsible.explain import global_importance, save_global_importance

            importance = global_importance(predictor.pipeline, train_texts)
            for path in save_global_importance(importance, output_root / "explain"):
                mlflow.log_artifact(str(path), artifact_path="explain")
            logger.info("logged SHAP global importance for %d classes", len(importance))

        fairness = measure_candidate_fairness(predictor)
        mlflow.log_metrics(fairness.as_metrics())
        fairness_path = output_root / "fairness_probe.json"
        from sentiment.responsible.fairness import write_report as write_fairness_report

        write_fairness_report(fairness, fairness_path)
        mlflow.log_artifact(str(fairness_path), artifact_path="fairness")
        logger.info("fairness max identity-pair delta %.6f", fairness.max_delta)

        result: Dict[str, Any] = dict(metrics)
        result["run_id"] = active_run.info.run_id
        result["fairness"] = fairness
        if cv_summary is not None:
            result["cross_validation"] = cv_summary
        if tuning is not None:
            result["tuning"] = tuning
        return result


def measure_candidate_fairness(predictor: Any) -> Any:
    """Probe a fitted candidate model in-process, before it is ever deployed.

    Two fairness measurements exist in this system and they are not
    interchangeable:

    * **This one** runs against a candidate that has not been promoted, so it can
      gate promotion. There is no service to query yet, so it calls the predictor
      directly.
    * ``scripts/run_fairness_probe.py`` runs over HTTP against the deployed
      model, so the number on the dashboard describes what users actually talk
      to, including its preprocessing.

    Both use the same probe set and the same threshold, so the gate and the alert
    cannot disagree about what counts as biased.
    """
    from sentiment.responsible.fairness import probe

    def score(texts: Any) -> list:
        return [prediction.score for prediction in predictor.predict(list(texts))]

    return probe(score)


def clean_text_vietnamese(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'doubledot', ':', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfraction\b', '/', text, flags=re.IGNORECASE)
    text = re.sub(r'wzjwz\d+', '[ANON]', text, flags=re.IGNORECASE)
    return text.strip()


def train_transformer_model(
    model_name: str = settings.model_name,
    dataset_name: str = settings.model_dataset_name,
    output_dir: str = "./artifacts/xlm-roberta",
    epochs: int = settings.epochs,
    batch_size: int = settings.batch_size,
    learning_rate: float = settings.learning_rate,
    max_length: int = settings.max_length,
    apply_cleaning: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fine-tune XLM-RoBERTa on UIT-VSFC and log experiments to MLflow."""
    from transformers import set_seed
    set_seed(seed)

    logger.info("Loading dataset '%s'...", dataset_name)
    ds = load_dataset(dataset_name)

    logger.info("Loading tokenizer and model for '%s'...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        texts = examples["Sentence"]
        if apply_cleaning:
            texts = [clean_text_vietnamese(t) for t in texts]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_ds = ds.map(tokenize_fn, batched=True)
    tokenized_ds = tokenized_ds.rename_column("Encoded_sentiment", "label")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=settings.num_labels,
        id2label=settings.label_map,
        label2id=settings.rev_label_map,
    )

    # Compute balanced weights so the minority neutral class contributes equally.
    label_counts = Counter(ds["train"]["Encoded_sentiment"])
    total = sum(label_counts.values())
    num_classes = settings.num_labels
    class_weights = torch.tensor(
        [total / (num_classes * label_counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float,
    )
    logger.info("Class weights: %s", class_weights.tolist())

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
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,  # Chỉ giữ 2 checkpoint gần nhất để tiết kiệm disk
        fp16=use_fp16,
        report_to=[],  # We manage MLflow manually via mlflow.start_run()
        dataloader_num_workers=2,
    )

    with mlflow.start_run(run_name=f"fine-tune-{model_name.replace('/', '-')}"):
        mlflow.set_tag("trained_at", datetime.now(timezone.utc).isoformat())
        mlflow.log_params(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_length": max_length,
            }
        )

        val_split = "validation" if "validation" in tokenized_ds else "dev"

        # Custom Trainer with class weights for NEUTRAL imbalance
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                # Use next(model.parameters()).device — works for both single GPU and DataParallel
                device = next(model.parameters()).device
                loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
                loss = loss_fn(logits, labels)
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds[val_split],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
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
        drift_reference = build_drift_reference(
            ds["train"]["Sentence"], ds["train"]["Encoded_sentiment"]
        )
        drift_path = Path(output_dir) / "drift_reference.json"
        drift_path.write_text(json.dumps(drift_reference.to_dict()), encoding="utf-8")
        mlflow.log_artifacts(output_dir, artifact_path="model")

        return dict(test_results)
