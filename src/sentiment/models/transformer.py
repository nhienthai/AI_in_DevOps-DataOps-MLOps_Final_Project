"""XLM-RoBERTa / Transformer Predictor for Vietnamese Sentiment Classification."""

import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.sentiment.config import settings
from src.sentiment/serving/predictor import Predictor


class TransformerPredictor(Predictor):
    """XLM-RoBERTa / HuggingFace Transformer Sentiment Predictor."""

    def __init__(
        self,
        model_name_or_path: str = settings.model_name,
        model_version: str = "xlm-roberta-v1.0",
        device: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.model_version = model_version
        self.label_map = settings.label_map
        self.rev_label_map = settings.rev_label_map
        self.max_length = settings.max_length

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = None
        self.model = None
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

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify single text sample."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a list of text strings in a single batched pass."""
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i, text in enumerate(texts):
            sample_probs = probs[i]
            pred_id = int(sample_probs.argmax())
            pred_label = self.label_map.get(pred_id, "UNKNOWN")
            confidence = float(sample_probs[pred_id])

            prob_dict = {
                self.label_map[idx]: round(float(prob), 4)
                for idx, prob in enumerate(sample_probs)
                if idx in self.label_map
            }

            results.append({
                "label": pred_label,
                "score": round(confidence, 4),
                "model_version": self.model_version,
                "probabilities": prob_dict,
            })

        return results

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
