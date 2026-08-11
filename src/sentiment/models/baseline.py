"""TF-IDF + Logistic Regression Baseline Predictor for sentiment classification."""

import os
from typing import Any, Dict, List, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.sentiment.config import settings
from src.sentiment.serving.predictor import Predictor


class BaselinePredictor(Predictor):
    """TF-IDF + Logistic Regression baseline model."""

    def __init__(self, model_version: str = "baseline-v1.0"):
        self.model_version = model_version
        self.label_map = settings.label_map
        self.rev_label_map = settings.rev_label_map
        self.pipeline: Optional[Pipeline] = None

    def fit(self, texts: List[str], labels: List[int]) -> "BaselinePredictor":
        """Train baseline TF-IDF + Logistic Regression pipeline."""
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        """Classify single text sample."""
        if self.pipeline is None:
            raise RuntimeError("Baseline model is not fitted yet.")

        probs = self.pipeline.predict_proba([text])[0]
        pred_label_id = int(self.pipeline.predict([text])[0])
        pred_label = self.label_map.get(pred_label_id, "UNKNOWN")
        confidence = float(probs[pred_label_id])

        prob_dict = {
            self.label_map[i]: float(prob)
            for i, prob in enumerate(probs)
            if i in self.label_map
        }

        return {
            "label": pred_label,
            "score": round(confidence, 4),
            "model_version": self.model_version,
            "probabilities": prob_dict,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify batch of text samples."""
        return [self.predict(text) for text in texts]

    def save(self, path: str) -> None:
        """Save model pipeline to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str, model_version: str = "baseline-v1.0") -> "BaselinePredictor":
        """Load model pipeline from file."""
        instance = cls(model_version=model_version)
        instance.pipeline = joblib.load(path)
        return instance
