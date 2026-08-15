#!/usr/bin/env python3
"""
Detailed model evaluation script for UIT-VSFC Sentiment Analysis.
Shows per-sample predictions, confusion matrix, and misclassified examples.

Usage on Kaggle:
    python scripts/evaluate_model.py \
        --model-path ./artifacts/xlm-roberta \
        --split test \
        --show-wrong 20 \
        --output-csv ./artifacts/eval_results.csv
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment.config import settings

LABEL_MAP = settings.label_map
LABEL_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]


def load_model(model_path: str, device: str):
    print(f"\n📦 Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict_batch(texts, tokenizer, model, device, max_length=128, batch_size=32):
    all_probs, all_preds = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
    return all_preds, all_probs


def print_confusion_matrix(y_true, y_pred, label_names):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    col_w = max(len(n) for n in label_names) + 2
    print("\n📊 Confusion Matrix (rows=Actual, cols=Predicted):")
    header = " " * col_w + "".join(f"{n:>{col_w}}" for n in label_names)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:<{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))


def print_classification_report(y_true, y_pred, label_names):
    from sklearn.metrics import classification_report

    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))


def show_misclassified(texts, y_true, y_pred, probs, label_names, n=20):
    wrong = [
        (i, texts[i], y_true[i], y_pred[i], probs[i])
        for i in range(len(texts))
        if y_true[i] != y_pred[i]
    ]
    print(f"\n❌ Misclassified ({min(n, len(wrong))} / {len(wrong)} total errors):")
    print("=" * 90)
    for rank, (_, text, true, pred, prob) in enumerate(wrong[:n], 1):
        print(
            f"#{rank:3d} | Actual: {label_names[true]:<10} | "
            f"Predicted: {label_names[pred]:<10} | Conf: {max(prob):.2%}"
        )
        print(f"      Text: {text[:120]}")
        print(f"      Probs: {' | '.join(f'{label_names[j]}={p:.2%}' for j, p in enumerate(prob))}")
        print()


def show_correct_samples(texts, y_true, y_pred, probs, label_names, n=10):
    correct = [(texts[i], y_true[i], probs[i]) for i in range(len(texts)) if y_true[i] == y_pred[i]]
    print(f"\n✅ Correct Predictions (first {n}):")
    print("=" * 90)
    for rank, (text, label, prob) in enumerate(correct[:n], 1):
        print(f"#{rank:3d} | Label: {label_names[label]:<10} | Conf: {max(prob):.2%}")
        print(f"      Text: {text[:120]}")
        print()


def save_csv(texts, y_true, y_pred, probs, label_names, output_csv: str):
    rows = []
    for i, text in enumerate(texts):
        row = {
            "text": text,
            "actual": label_names[y_true[i]],
            "predicted": label_names[y_pred[i]],
            "correct": y_true[i] == y_pred[i],
            "confidence": round(max(probs[i]), 4),
        }
        for j, name in enumerate(label_names):
            row[f"prob_{name}"] = round(probs[i][j], 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 Full results saved → {output_csv}")
    print(f"   Total: {len(df)} | Correct: {df['correct'].sum()} | Wrong: {(~df['correct']).sum()}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detailed Evaluation for Sentiment Model on UIT-VSFC"
    )
    parser.add_argument("--model-path", default="./artifacts/xlm-roberta")
    parser.add_argument("--dataset", default=settings.model_dataset_name)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--max-length", type=int, default=settings.max_length)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--show-wrong", type=int, default=20)
    parser.add_argument("--show-correct", type=int, default=10)
    parser.add_argument("--output-csv", default="./artifacts/eval_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device.upper()}")

    print(f"\n📂 Loading '{args.dataset}' [{args.split}]...")
    ds = load_dataset(args.dataset)
    split_data = ds[args.split]
    texts = list(split_data["Sentence"])
    y_true = list(split_data["Encoded_sentiment"])
    print(f"   Samples: {len(texts)}")
    for label_id, count in sorted(Counter(y_true).items()):
        print(f"   {LABEL_NAMES[label_id]}: {count} ({count/len(texts):.1%})")

    tokenizer, model = load_model(args.model_path, device)
    print(f"\n🔮 Running inference on {len(texts)} samples...")
    y_pred, probs = predict_batch(texts, tokenizer, model, device, args.max_length, args.batch_size)

    print_confusion_matrix(y_true, y_pred, LABEL_NAMES)
    print_classification_report(y_true, y_pred, LABEL_NAMES)
    show_misclassified(texts, y_true, y_pred, probs, LABEL_NAMES, args.show_wrong)
    show_correct_samples(texts, y_true, y_pred, probs, LABEL_NAMES, args.show_correct)
    save_csv(texts, y_true, y_pred, probs, LABEL_NAMES, args.output_csv)


if __name__ == "__main__":
    main()
