"""XLM-RoBERTa / Transformer Predictor for Vietnamese Sentiment Classification."""

import os
from collections.abc import Sequence
from typing import Any, List, Optional, cast

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment.config import settings
from sentiment.serving.predictor import Prediction, Predictor, SentimentLabel


class TransformerPredictor(Predictor):
    """XLM-RoBERTa / HuggingFace Transformer Sentiment Predictor."""

    def __init__(
        self,
        model_name_or_path: str = settings.model_name,
        model_version: str = "xlm-roberta-v1.0",
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.version = model_version
        self.label_map = settings.label_map
        self.rev_label_map = settings.rev_label_map
        self.max_length = settings.max_length
        self.stage = "Development"
        self.metrics: dict[str, float] = {}
        self.fairness_delta: float | None = None
        self.trained_at: str | None = None
        self.run_id: str | None = None

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer: Any = None
        self.model: Any = None
        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model architecture."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            num_labels=settings.num_labels,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a list of text strings in a single batched pass."""
        if not texts:
            return []

        original_lengths = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_length=True,
        )["length"]
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        results: list[Prediction] = []
        for i, _text in enumerate(texts):
            sample_probs = probs[i]
            pred_id = int(sample_probs.argmax())
            pred_label = self.label_map.get(pred_id)
            if pred_label not in {"negative", "neutral", "positive"}:
                raise RuntimeError(f"Model returned unsupported label id {pred_id}.")
            label = cast(SentimentLabel, pred_label)
            confidence = float(sample_probs[pred_id])
            positive_id = self.rev_label_map["positive"]
            positive_score = float(sample_probs[positive_id])

            results.append(
                Prediction(
                    label=label,
                    score=positive_score,
                    confidence=confidence,
                    truncated=int(original_lengths[i]) > self.max_length,
                )
            )

        return results

    def predict_batch(self, texts: List[str]) -> list[Prediction]:
        """Compatibility alias for training and validation callers."""
        return self.predict(texts)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and tokenizer weights locally."""
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls,
        save_directory: str,
        model_version: str = "xlm-roberta-v1.0",
        device: Optional[str] = None,
    ) -> "TransformerPredictor":
        """Instantiate TransformerPredictor from local weights directory."""
        return cls(
            model_name_or_path=save_directory,
            model_version=model_version,
            device=device,
        )
