"""TF-IDF + Logistic Regression Baseline Predictor for sentiment classification."""

import os
from collections.abc import Sequence
from typing import List, Optional, cast

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentiment.config import settings
from sentiment.serving.predictor import Prediction, Predictor, SentimentLabel


class BaselinePredictor(Predictor):
    """TF-IDF + Logistic Regression baseline model."""

    def __init__(self, model_version: str = "baseline-v1.0", blind_identity: bool = False):
        """Build an unfitted baseline predictor.

        Args:
            model_version: Version string reported through ``/model/info``.
            blind_identity: Replace identity terms with a neutral placeholder
                before vectorising. Applied in both ``fit`` and ``predict``, so a
                blinded model can never be served unblinded text.
        """
        self.version = model_version
        self.blind_identity = blind_identity
        self.label_map = settings.label_map
        self.rev_label_map = settings.rev_label_map
        self.pipeline: Optional[Pipeline] = None
        self.stage = "Development"
        self.metrics: dict[str, float] = {}
        self.fairness_delta: float | None = None
        self.trained_at: str | None = None
        self.run_id: str | None = None

    def _prepare(self, texts: Sequence[str]) -> List[str]:
        """Apply identity blinding when enabled."""
        if not self.blind_identity:
            return list(texts)
        from sentiment.responsible.fairness import blind_identity_terms

        return [blind_identity_terms(text) for text in texts]

    def fit(
        self,
        texts: List[str],
        labels: List[int],
        tfidf_params: Optional[dict] = None,
        clf_params: Optional[dict] = None,
    ) -> "BaselinePredictor":
        """Train the baseline TF-IDF + Logistic Regression pipeline.

        Args:
            texts: Training texts.
            labels: Integer labels aligned with ``texts``.
            tfidf_params: Overrides for the vectoriser, as produced by the
                Optuna sweep in :mod:`sentiment.training.tune`.
            clf_params: Overrides for the classifier.

        Returns:
            This predictor, fitted.
        """
        vectoriser_kwargs = {"ngram_range": (1, 2), "max_features": 10000}
        vectoriser_kwargs.update(tfidf_params or {})
        classifier_kwargs = {
            "max_iter": 1000,
            "C": 1.0,
            "random_state": 42,
            "class_weight": "balanced",
        }
        classifier_kwargs.update(clf_params or {})

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vectoriser_kwargs)),
                ("clf", LogisticRegression(**classifier_kwargs)),
            ]
        )
        self.pipeline.fit(self._prepare(texts), labels)
        return self

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch of text samples."""
        if self.pipeline is None:
            raise RuntimeError("Baseline model is not fitted yet.")

        prepared = self._prepare(texts)
        probabilities = self.pipeline.predict_proba(prepared)
        label_ids = self.pipeline.predict(prepared)
        results: list[Prediction] = []
        for label_id, probs in zip(label_ids, probabilities):
            class_index = list(self.pipeline.classes_).index(label_id)
            confidence = float(probs[class_index])
            positive_id = self.rev_label_map["positive"]
            positive_score = (
                float(probs[list(self.pipeline.classes_).index(positive_id)])
                if positive_id in self.pipeline.classes_
                else 0.0
            )
            label = self.label_map.get(int(label_id), "neutral")
            if label not in {"negative", "neutral", "positive"}:
                raise RuntimeError(f"Model returned unsupported label id {label_id}.")
            results.append(
                Prediction(
                    label=cast(SentimentLabel, label),
                    score=positive_score,
                    confidence=confidence,
                    truncated=False,
                )
            )
        return results

    def predict_batch(self, texts: List[str]) -> list[Prediction]:
        """Classify batch of text samples."""
        return self.predict(texts)

    def save(self, path: str) -> None:
        """Save the pipeline together with the preprocessing it was fitted under.

        Saving the pipeline alone loses ``blind_identity``, and a blinded model
        reloaded without it silently predicts on text it was never fitted on —
        the exact train/serve skew blinding is supposed to prevent. The flag
        therefore travels inside the artifact rather than beside it.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "format": 2,
                "pipeline": self.pipeline,
                "blind_identity": self.blind_identity,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, model_version: str = "baseline-v1.0") -> "BaselinePredictor":
        """Load a saved model, restoring the preprocessing it was fitted under.

        Accepts artifacts written before the flag was persisted: a bare pipeline
        is treated as unblinded, which is what it was.
        """
        payload = joblib.load(path)
        if isinstance(payload, dict) and "pipeline" in payload:
            instance = cls(
                model_version=model_version,
                blind_identity=bool(payload.get("blind_identity", False)),
            )
            instance.pipeline = payload["pipeline"]
        else:
            instance = cls(model_version=model_version)
            instance.pipeline = payload
        return instance
