"""The prediction interface protocol and a deterministic stub predictor."""

from typing import Dict, List, Protocol, Any
from src.sentiment.config import settings


class Predictor(Protocol):
    def predict(self, text: str) -> Dict[str, Any]:
        """Classify a single text string."""
        ...

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify a batch of text strings."""
        ...


class StubPredictor:
    """Deterministic stub predictor for testing serving infrastructure."""

    def __init__(self, model_version: str = "stub-v1.0"):
        self.model_version = model_version
        self.label_map = settings.label_map

    def predict(self, text: str) -> Dict[str, Any]:
        cleaned = text.lower()
        if any(w in cleaned for w in ["tốt", "hay", "đầy đủ", "xuất sắc", "thích", "great", "good"]):
            label = "POSITIVE"
            score = 0.95
        elif any(w in cleaned for w in ["dở", "tệ", "kém", "chưa", "chán", "bad", "poor"]):
            label = "NEGATIVE"
            score = 0.91
        else:
            label = "NEUTRAL"
            score = 0.80

        return {
            "label": label,
            "score": score,
            "model_version": self.model_version,
            "probabilities": {
                "NEGATIVE": 0.91 if label == "NEGATIVE" else 0.05,
                "NEUTRAL": 0.80 if label == "NEUTRAL" else 0.05,
                "POSITIVE": 0.95 if label == "POSITIVE" else 0.05,
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.predict(text) for text in texts]
