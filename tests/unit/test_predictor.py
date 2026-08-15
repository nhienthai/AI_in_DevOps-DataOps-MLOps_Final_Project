"""Predictor interface and stub behaviour tests."""

from dataclasses import FrozenInstanceError

import pytest

from sentiment.serving.predictor import Prediction, Predictor, StubPredictor, validate_predictions


def test_stub_satisfies_predictor_protocol() -> None:
    assert isinstance(StubPredictor(), Predictor)


def test_stub_is_deterministic_and_returns_one_result_per_text() -> None:
    predictor = StubPredictor()
    first = predictor.predict(["hello", "world"])
    second = predictor.predict(["hello", "world"])
    assert first == second
    assert len(first) == 2


def test_label_and_confidence_are_derived_from_score() -> None:
    for result in StubPredictor().predict([f"review {index}" for index in range(50)]):
        assert result.label == ("positive" if result.score >= 0.5 else "negative")
        assert result.confidence == pytest.approx(max(result.score, 1 - result.score))
        assert 0.5 <= result.confidence <= 1.0


def test_text_is_hashed_as_utf8_and_long_input_is_truncated() -> None:
    results = StubPredictor(max_chars=5).predict(["ดีมาก 😊", "short"])
    assert results[0].truncated is True
    assert results[1].truncated is False


def test_prediction_is_immutable() -> None:
    result = Prediction(label="positive", score=0.9, confidence=0.9, truncated=False)
    with pytest.raises(FrozenInstanceError):
        result.score = 0.1  # type: ignore[misc]


@pytest.mark.parametrize(
    "predictions",
    [
        [],
        [Prediction("unsupported", 0.5, 0.5, False)],
        [Prediction("positive", 1.1, 0.9, False)],
        [Prediction("positive", 0.9, -0.1, False)],
    ],
)
def test_predictor_output_validation_rejects_malformed_results(
    predictions: list[Prediction],
) -> None:
    with pytest.raises(ValueError):
        validate_predictions(predictions, 1)
