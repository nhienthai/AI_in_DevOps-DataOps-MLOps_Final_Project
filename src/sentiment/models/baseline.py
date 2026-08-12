"""TF-IDF + Logistic Regression Baseline Predictor for sentiment classification."""

import os
from collections.abc import Sequence
from typing import List, Optional
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment.config import settings
from sentiment.serving.predictor import Prediction, Predictor


class BaselinePredictor(Predictor):
    """TF-IDF + Logistic Regression baseline model."""

    def __init__(self, model_version: str = "baseline-v1.0"):
        self.version = model_version
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

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch of text samples."""
        if self.pipeline is None:
            raise RuntimeError("Baseline model is not fitted yet.")

        probabilities = self.pipeline.predict_proba(list(texts))
        label_ids = self.pipeline.predict(list(texts))
        results: list[Prediction] = []
        for label_id, probs in zip(label_ids, probabilities):
            class_index = list(self.pipeline.classes_).index(label_id)
            confidence = float(probs[class_index])
            results.append(
                Prediction(
                    label=self.label_map.get(int(label_id), "unknown"),
                    score=confidence,
                    confidence=confidence,
                    truncated=False,
                )
            )
        return results

    def predict_batch(self, texts: List[str]) -> list[Prediction]:
        """Classify batch of text samples."""
        return self.predict(texts)

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
