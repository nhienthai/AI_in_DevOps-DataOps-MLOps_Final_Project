"""LIME local explanations and SHAP global importance."""

from pathlib import Path

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("lime")

from sentiment.models.baseline import BaselinePredictor  # noqa: E402
from sentiment.responsible.explain import (  # noqa: E402
    LimeExplainer,
    _as_per_class,
    global_importance,
    save_global_importance,
)

TEXTS = [
    "thầy dạy rất hay và nhiệt tình",
    "bài giảng rất dễ hiểu",
    "giáo trình đầy đủ chi tiết",
    "bài giảng nhàm chán khó hiểu",
    "giáo viên đi quá nhanh",
    "tài liệu sơ sài không có ví dụ",
    "phòng học bình thường",
    "môn học có ba tín chỉ",
    "lịch học vào thứ hai",
]
LABELS = [2, 2, 2, 0, 0, 0, 1, 1, 1]


@pytest.fixture(scope="module")
def fitted() -> BaselinePredictor:
    return BaselinePredictor().fit(TEXTS, LABELS)


def test_lime_returns_the_public_contract(fitted: BaselinePredictor) -> None:
    explainer = LimeExplainer(fitted, num_features=4, num_samples=60)
    explanation = explainer.explain("thầy dạy rất hay")

    assert explanation.method == "lime"
    assert explanation.label in {"negative", "neutral", "positive"}
    assert 0.0 <= explanation.score <= 1.0
    assert 1 <= len(explanation.attributions) <= 4
    assert all(isinstance(item.token, str) for item in explanation.attributions)
    assert all(isinstance(item.attribution, float) for item in explanation.attributions)


def test_lime_is_deterministic_for_the_same_input(fitted: BaselinePredictor) -> None:
    """A seeded explainer must not give a different story on a second look."""
    first = LimeExplainer(fitted, num_features=4, num_samples=60).explain("thầy dạy rất hay")
    second = LimeExplainer(fitted, num_features=4, num_samples=60).explain("thầy dạy rất hay")
    assert [item.token for item in first.attributions] == [
        item.token for item in second.attributions
    ]


def test_lime_label_matches_the_predictor(fitted: BaselinePredictor) -> None:
    text = "bài giảng nhàm chán khó hiểu"
    explanation = LimeExplainer(fitted, num_features=3, num_samples=60).explain(text)
    assert explanation.label == fitted.predict([text])[0].label


def test_unsupported_method_raises(fitted: BaselinePredictor) -> None:
    with pytest.raises(ValueError, match="unsupported explanation method"):
        LimeExplainer(fitted).explain("thầy dạy hay", method="shap")  # type: ignore[arg-type]


def test_blank_text_raises(fitted: BaselinePredictor) -> None:
    with pytest.raises(ValueError, match="empty text"):
        LimeExplainer(fitted).explain("   ")


def test_shap_output_shapes_are_normalised() -> None:
    """The three-dimensional multiclass array is the shape that silently breaks."""
    import numpy as np

    three_d = np.zeros((4, 5, 3))
    per_class = _as_per_class(three_d, n_classes=3)
    assert len(per_class) == 3
    assert all(item.shape == (4, 5) for item in per_class)

    as_list = [np.zeros((4, 5))] * 3
    assert len(_as_per_class(as_list, n_classes=3)) == 3

    binary = np.zeros((4, 5))
    assert len(_as_per_class(binary, n_classes=2)) == 2

    with pytest.raises(ValueError, match="cannot split per class"):
        _as_per_class(binary, n_classes=3)

    with pytest.raises(ValueError, match="unexpected SHAP output"):
        _as_per_class(np.zeros(4), n_classes=2)


def test_global_importance_covers_every_class(fitted: BaselinePredictor) -> None:
    pytest.importorskip("shap")
    importance = global_importance(fitted.pipeline, TEXTS, top_k=3)

    assert set(importance) == {"negative", "neutral", "positive"}
    for features in importance.values():
        assert len(features) <= 3
        assert all("feature" in item and "mean_abs_shap" in item for item in features)


def test_global_importance_rejects_a_foreign_pipeline() -> None:
    pytest.importorskip("shap")

    class NotAPipeline:
        named_steps: dict = {}

    with pytest.raises(ValueError, match="tfidf"):
        global_importance(NotAPipeline(), TEXTS)


def test_save_global_importance_writes_json_and_charts(
    fitted: BaselinePredictor, tmp_path: Path
) -> None:
    pytest.importorskip("shap")
    importance = global_importance(fitted.pipeline, TEXTS, top_k=3)
    written = save_global_importance(importance, tmp_path / "explain")

    assert (tmp_path / "explain" / "shap_global_importance.json").exists()
    assert any(path.suffix == ".png" for path in written)
