"""Model behaviour tests that need a real dataset or a real checkpoint.

Marked ``slow`` and excluded from every push: they download UIT-VSFC and, for the
transformer, roughly a gigabyte of weights. The nightly workflow runs them.

These assert behaviour rather than plumbing — known-positive and known-negative
cases, calibration bounds, and the CPU latency budget. The fast suite covers the
contracts; this covers whether the model is any good.
"""

import time

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("datasets")

from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.serving.predictor import validate_predictions  # noqa: E402

pytestmark = pytest.mark.slow

KNOWN_POSITIVE = [
    "thầy dạy rất hay và nhiệt tình",
    "bài giảng dễ hiểu , ví dụ sinh động",
    "môn học rất bổ ích và thú vị",
    "giảng viên nhiệt tình giải đáp thắc mắc",
]

KNOWN_NEGATIVE = [
    "bài giảng nhàm chán và khó hiểu",
    "giảng viên dạy quá nhanh , không theo kịp",
    "tài liệu sơ sài , không có ví dụ",
    "cách chấm điểm không rõ ràng",
]


@pytest.fixture(scope="module")
def trained() -> BaselinePredictor:
    """Train the baseline on the real corpus once for the whole module."""
    from datasets import load_dataset

    from sentiment.config import settings

    dataset = load_dataset(settings.dataset_name)
    texts = list(dataset["train"]["Sentence"])
    labels = [int(value) for value in dataset["train"]["Encoded_sentiment"]]
    return BaselinePredictor().fit(texts, labels)


def test_known_positive_sentences_are_not_negative(trained: BaselinePredictor) -> None:
    predictions = trained.predict(KNOWN_POSITIVE)
    labels = [prediction.label for prediction in predictions]
    assert labels.count("positive") >= 3, f"expected mostly positive, got {labels}"
    assert "negative" not in labels


def test_known_negative_sentences_are_not_positive(trained: BaselinePredictor) -> None:
    predictions = trained.predict(KNOWN_NEGATIVE)
    labels = [prediction.label for prediction in predictions]
    assert labels.count("negative") >= 3, f"expected mostly negative, got {labels}"
    assert "positive" not in labels


def test_positive_scores_exceed_negative_scores(trained: BaselinePredictor) -> None:
    """The continuous output must order the two groups, not merely label them."""
    positive = [item.score for item in trained.predict(KNOWN_POSITIVE)]
    negative = [item.score for item in trained.predict(KNOWN_NEGATIVE)]
    assert min(positive) > max(negative)


def test_calibration_bounds(trained: BaselinePredictor) -> None:
    """Confidence must be a probability and never below chance for three classes."""
    predictions = trained.predict(KNOWN_POSITIVE + KNOWN_NEGATIVE)
    validate_predictions(predictions, len(KNOWN_POSITIVE) + len(KNOWN_NEGATIVE))

    for prediction in predictions:
        assert 1 / 3 <= prediction.confidence <= 1.0
        assert 0.0 <= prediction.score <= 1.0

    confidences = [prediction.confidence for prediction in predictions]
    assert max(confidences) < 1.0, "a perfectly certain model is a miscalibrated one"


def test_held_out_quality_meets_the_promotion_floor(trained: BaselinePredictor) -> None:
    """The same thresholds validate_model.py enforces, checked on the test split."""
    from datasets import load_dataset

    from sentiment.config import settings
    from sentiment.training.evaluate import evaluate_predictions

    dataset = load_dataset(settings.dataset_name)
    texts = list(dataset["test"]["Sentence"])
    labels = [int(value) for value in dataset["test"]["Encoded_sentiment"]]

    predicted = [settings.rev_label_map[prediction.label] for prediction in trained.predict(texts)]
    metrics = evaluate_predictions(labels, predicted)

    assert metrics["macro_f1"] >= 0.70, f"macro-F1 regressed to {metrics['macro_f1']:.4f}"
    assert metrics["accuracy"] >= 0.85, f"accuracy regressed to {metrics['accuracy']:.4f}"


def test_latency_budget_on_cpu(trained: BaselinePredictor) -> None:
    """N1: p95 under 200 ms on CPU, measured rather than assumed."""
    from sentiment.training.evaluate import check_latency_budget

    result = check_latency_budget(trained, p95_target_ms=200.0, num_runs=50)
    assert result["passed_sla"], f"p95 was {result['p95_latency_ms']} ms"


def test_fairness_gate_holds_after_blinding() -> None:
    """The mitigation that ships must keep delivering parity on real data."""
    from datasets import load_dataset

    from sentiment.config import settings
    from sentiment.training.train import measure_candidate_fairness

    dataset = load_dataset(settings.dataset_name)
    texts = list(dataset["train"]["Sentence"])
    labels = [int(value) for value in dataset["train"]["Encoded_sentiment"]]

    blinded = BaselinePredictor(blind_identity=True).fit(texts, labels)
    result = measure_candidate_fairness(blinded)
    assert result.passes(0.10), f"max identity-pair delta was {result.max_delta:.6f}"


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch is required"),
    reason="torch unavailable",
)
def test_transformer_checkpoint_satisfies_the_serving_contract() -> None:
    """A real checkpoint must load and answer within the published contract.

    The head is untrained here, so this asserts shape and ranges rather than
    accuracy — enough to catch a transformers upgrade breaking the loader.
    """
    pytest.importorskip("transformers")
    from sentiment.models.transformer import TransformerPredictor

    predictor = TransformerPredictor.from_pretrained("xlm-roberta-base")

    started = time.perf_counter()
    predictions = predictor.predict(["thầy dạy rất hay", "bài giảng nhàm chán"])
    elapsed_ms = (time.perf_counter() - started) * 1000

    validate_predictions(predictions, 2)
    for prediction in predictions:
        assert prediction.label in {"negative", "neutral", "positive"}

    assert elapsed_ms < 10_000, f"two predictions took {elapsed_ms:.0f} ms"
