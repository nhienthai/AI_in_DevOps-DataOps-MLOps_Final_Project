"""Fairness probe, mitigations, and the report round-trip.

The probe is a measuring instrument, so these tests check it against known
answers: a scorer with a deliberate, known bias must produce that bias on the
right axis, and a scorer with none must produce zero.
"""

from pathlib import Path

import pytest

from sentiment.responsible.fairness import (
    GROUPS,
    TEMPLATES,
    blind_identity_terms,
    build_sentences,
    counterfactual_augment,
    counterfactual_pairs,
    identity_terms,
    load_report,
    probe,
    write_report,
)


def constant(value: float):
    """Return a scorer that ignores its input."""

    def score(texts):
        return [value] * len(texts)

    return score


def test_probe_set_covers_every_template_and_term() -> None:
    rows = build_sentences()
    expected = len(TEMPLATES) * sum(len(group.terms) for group in GROUPS)
    assert len(rows) == expected
    assert all("{person}" not in sentence for _, _, _, sentence in rows)


def test_unbiased_scorer_has_zero_delta() -> None:
    result = probe(constant(0.7))
    assert result.max_delta == pytest.approx(0.0)
    assert result.mean_delta == pytest.approx(0.0)
    assert result.passes(0.10)


def test_known_bias_is_detected_on_the_right_axis() -> None:
    def gendered(texts):
        return [0.9 if "thầy" in text else 0.5 for text in texts]

    result = probe(gendered)
    assert result.max_delta > 0.10
    assert not result.passes(0.10)
    assert result.max_delta_by_dimension["gender"] > 0.0
    assert result.max_delta_by_dimension["seniority"] == pytest.approx(0.0)


def test_only_same_dimension_groups_are_compared() -> None:
    """Comparing a gender term against a seniority term would be meaningless."""
    result = probe(constant(0.5))
    dimensions = {group.dimension for group in GROUPS}
    assert set(result.max_delta_by_dimension) <= dimensions


def test_wrong_number_of_scores_raises() -> None:
    with pytest.raises(ValueError, match="score_fn returned"):
        probe(lambda texts: [0.5])


def test_identity_terms_are_longest_first() -> None:
    """Short terms must not be substituted inside longer ones."""
    terms = identity_terms()
    assert terms == sorted(terms, key=len, reverse=True)


def test_blinding_removes_every_identity_term() -> None:
    blinded = blind_identity_terms("thầy người hà nội dạy rất hay")
    assert "thầy" not in blinded
    assert "hà nội" not in blinded
    assert "dạy rất hay" in blinded


def test_blinding_makes_an_identity_pair_identical() -> None:
    """This is why blinding drives the delta to exactly zero."""
    left = blind_identity_terms("thầy dạy rất nhiệt tình .")
    right = blind_identity_terms("cô dạy rất nhiệt tình .")
    assert left == right


def test_blinding_is_case_insensitive() -> None:
    assert blind_identity_terms("Thầy dạy hay") == blind_identity_terms("thầy dạy hay")


def test_counterfactual_pairs_stay_within_a_dimension() -> None:
    dimension_of = {term: group.dimension for group in GROUPS for term in group.terms}
    for source, target in counterfactual_pairs():
        assert dimension_of[source] == dimension_of[target]


def test_augmentation_preserves_labels_and_skips_untouched_rows() -> None:
    texts = ["thầy dạy rất hay", "phòng học bình thường"]
    labels = [2, 1]
    out_texts, out_labels = counterfactual_augment(texts, labels)

    assert out_texts[:2] == texts
    assert len(out_texts) == 3
    assert out_labels[2] == 2
    assert "thầy" not in out_texts[2]


def test_augmentation_is_case_insensitive() -> None:
    out_texts, _ = counterfactual_augment(["Thầy dạy hay"], [2])
    assert len(out_texts) == 2


def test_augmentation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="differ"):
        counterfactual_augment(["a", "b"], [1])


def test_report_round_trips(tmp_path: Path) -> None:
    result = probe(constant(0.6))
    path = write_report(result, tmp_path / "reports" / "fairness.json")
    restored = load_report(path)

    assert restored.n_sentences == result.n_sentences
    assert restored.n_pairs == result.n_pairs
    assert restored.max_delta == pytest.approx(result.max_delta)
    assert restored.max_delta_by_dimension == result.max_delta_by_dimension


def test_metrics_are_flat_floats_for_mlflow() -> None:
    metrics = probe(constant(0.5)).as_metrics()
    assert "fairness_max_delta" in metrics
    assert any(key.startswith("fairness_group_score_") for key in metrics)
    assert all(isinstance(value, float) for value in metrics.values())
