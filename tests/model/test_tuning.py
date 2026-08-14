"""Cross-validation, the Optuna sweep, and identity blinding at fit/predict time."""

import pytest

pytest.importorskip("sklearn")

from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.training.evaluate import cross_validate_baseline  # noqa: E402

POSITIVE = ["thầy dạy rất hay", "bài giảng dễ hiểu", "giáo trình đầy đủ", "rất bổ ích"]
NEGATIVE = ["bài giảng nhàm chán", "giảng quá nhanh", "tài liệu sơ sài", "rất khó theo"]
NEUTRAL = ["phòng học bình thường", "môn ba tín chỉ", "học thứ hai", "lớp bốn mươi người"]

TEXTS = POSITIVE * 3 + NEGATIVE * 3 + NEUTRAL * 3
LABELS = [2] * 12 + [0] * 12 + [1] * 12


def test_cross_validation_reports_mean_and_spread() -> None:
    summary = cross_validate_baseline(TEXTS, LABELS, n_splits=3)

    assert summary["n_splits"] == 3
    assert len(summary["folds"]) == 3
    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        assert 0.0 <= summary[f"cv_{metric}_mean"] <= 1.0
        assert summary[f"cv_{metric}_std"] >= 0.0


def test_cross_validation_is_reproducible() -> None:
    """Without a fixed seed the fold split moves and comparisons become noise."""
    first = cross_validate_baseline(TEXTS, LABELS, n_splits=3, seed=7)
    second = cross_validate_baseline(TEXTS, LABELS, n_splits=3, seed=7)
    assert first["cv_macro_f1_mean"] == pytest.approx(second["cv_macro_f1_mean"])


def test_cross_validation_honours_hyperparameters() -> None:
    summary = cross_validate_baseline(
        TEXTS,
        LABELS,
        n_splits=3,
        tfidf_params={"max_features": 50},
        clf_params={"C": 0.1},
    )
    assert "cv_macro_f1_mean" in summary


def test_cross_validation_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="differ"):
        cross_validate_baseline(TEXTS, LABELS[:5], n_splits=3)


def test_optuna_sweep_explores_and_returns_the_best() -> None:
    pytest.importorskip("optuna")
    from sentiment.training.tune import tune_baseline

    result = tune_baseline(TEXTS, LABELS, n_trials=3, n_splits=3, log_to_mlflow=False)

    assert result["n_trials"] == 3
    assert len(result["trials"]) == 3
    assert 0.0 <= result["best_value"] <= 1.0
    assert "ngram_range" in result["best_tfidf_params"]
    assert "C" in result["best_clf_params"]
    assert result["best_value"] == pytest.approx(
        max(trial["cv_macro_f1_mean"] for trial in result["trials"])
    )


def test_optuna_sweep_is_reproducible() -> None:
    pytest.importorskip("optuna")
    from sentiment.training.tune import tune_baseline

    first = tune_baseline(TEXTS, LABELS, n_trials=3, n_splits=3, seed=11, log_to_mlflow=False)
    second = tune_baseline(TEXTS, LABELS, n_trials=3, n_splits=3, seed=11, log_to_mlflow=False)
    assert first["best_params"] == second["best_params"]


def test_baseline_accepts_tuned_hyperparameters() -> None:
    model = BaselinePredictor().fit(
        TEXTS, LABELS, tfidf_params={"max_features": 40}, clf_params={"C": 2.0}
    )
    assert model.pipeline.named_steps["tfidf"].max_features == 40
    assert model.pipeline.named_steps["clf"].C == 2.0


def test_blinding_applies_to_prediction_not_only_training() -> None:
    """A blinded model served unblinded text sees a distribution it never saw."""
    model = BaselinePredictor(blind_identity=True).fit(TEXTS, LABELS)
    left = model.predict(["thầy dạy rất hay"])[0]
    right = model.predict(["cô dạy rất hay"])[0]

    assert left.label == right.label
    assert left.score == pytest.approx(right.score)


def test_unblinded_model_may_distinguish_identity_terms() -> None:
    """The control for the test above: without blinding the terms reach the model."""
    model = BaselinePredictor(blind_identity=False).fit(TEXTS, LABELS)
    vocabulary = model.pipeline.named_steps["tfidf"].vocabulary_
    assert "thầy" in vocabulary


def test_blinding_keeps_identity_terms_out_of_the_vocabulary() -> None:
    model = BaselinePredictor(blind_identity=True).fit(TEXTS, LABELS)
    vocabulary = model.pipeline.named_steps["tfidf"].vocabulary_
    assert "thầy" not in vocabulary


def test_blinding_survives_a_save_load_round_trip(tmp_path) -> None:
    """Regression: a blinded model reloaded unblinded predicts on unseen text.

    The fairness gate caught this in practice — a model trained with blinding
    measured a 0.08 identity gap after a joblib round-trip instead of 0.00.
    """
    model = BaselinePredictor(blind_identity=True).fit(TEXTS, LABELS)
    path = tmp_path / "model" / "baseline.joblib"
    model.save(str(path))

    restored = BaselinePredictor.load(str(path))
    assert restored.blind_identity is True

    left = restored.predict(["thầy dạy rất hay"])[0]
    right = restored.predict(["cô dạy rất hay"])[0]
    assert left.score == pytest.approx(right.score)


def test_loading_a_legacy_bare_pipeline_still_works(tmp_path) -> None:
    """Artifacts written before the flag existed were unblinded, and load as such."""
    import joblib

    model = BaselinePredictor().fit(TEXTS, LABELS)
    path = tmp_path / "legacy.joblib"
    joblib.dump(model.pipeline, path)

    restored = BaselinePredictor.load(str(path))
    assert restored.blind_identity is False
    assert restored.predict(["thầy dạy rất hay"])
