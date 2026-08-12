"""Prediction and explanation interfaces used by the HTTP serving layer."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """One sentiment result produced by a predictor."""

    label: str
    score: float
    confidence: float
    truncated: bool


@dataclass(frozen=True)
class TokenAttribution:
    """Contribution assigned to one token by an explanation method."""

    token: str
    attribution: float


@dataclass(frozen=True)
class Explanation:
    """Local explanation for a prediction."""

    method: Literal["lime"]
    label: str
    score: float
    attributions: tuple[TokenAttribution, ...]


@runtime_checkable
class Predictor(Protocol):
    """Structural interface implemented by every served model."""

    version: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Return one prediction for each input text."""
        ...


@runtime_checkable
class Explainer(Protocol):
    """Interface M5's LIME explainer plugs into."""

    def explain(self, text: str, method: Literal["lime"] = "lime") -> Explanation:
        """Explain one deployed-model decision."""
        ...


def _to_prediction(score: float, truncated: bool) -> Prediction:
    """Build a consistent binary prediction from a positive-class probability."""
    bounded_score = min(max(score, 0.0), 1.0)
    return Prediction(
        label="positive" if bounded_score >= 0.5 else "negative",
        score=bounded_score,
        confidence=max(bounded_score, 1.0 - bounded_score),
        truncated=truncated,
    )


class StubPredictor:
    """Deterministic placeholder used until the registry model is configured."""

    version = "stub-0"
    stage = "Development"
    metrics: dict[str, float] = {}
    fairness_delta: float | None = None
    trained_at: str | None = None
    run_id: str | None = None

    def __init__(self, max_chars: int = 5_000) -> None:
        self._max_chars = max_chars

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Hash each text into a reproducible score in ``[0, 1]``."""
        predictions: list[Prediction] = []
        for text in texts:
            truncated = len(text) > self._max_chars
            digest = hashlib.sha256(text[: self._max_chars].encode("utf-8")).digest()
            score = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
            predictions.append(_to_prediction(score, truncated))
        return predictions
