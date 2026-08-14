import pandas as pd
import pytest

from sentiment.data.validate import DataQualityError, check, validate


def frame(n: int = 2000, neutral_share: float = 0.1) -> pd.DataFrame:
    """Build a three-class frame with a controllable minority-class share.

    ``neutral`` is the minority class in UIT-VSFC, so it is the one the gate has
    to protect: a split that loses it cannot train or score that class.
    """
    n_neutral = int(n * neutral_share)
    n_rest = n - n_neutral
    labels = [2] * (n_rest // 2) + [0] * (n_rest - n_rest // 2) + [1] * n_neutral
    return pd.DataFrame({"label": labels, "text": [f"review number {i}" for i in range(n)]})


def test_clean_frame_passes():
    report = check(frame())
    assert report.passed
    assert report.failures == ()
    assert report.n_rows == 2000


def test_missing_column_fails():
    report = check(pd.DataFrame({"label": [1] * 2000}))
    assert not report.passed
    assert any("column" in f for f in report.failures)


def test_empty_text_fails():
    df = frame()
    df.loc[0, "text"] = "   "
    report = check(df)
    assert not report.passed
    assert report.n_empty_text == 1


def test_too_few_rows_fails():
    assert not check(frame(n=10)).passed


def test_duplicate_rate_above_threshold_fails():
    df = frame()
    df.loc[:200, "text"] = "identical review"
    report = check(df)
    assert not report.passed
    assert report.n_duplicates >= 200


def test_minority_class_below_floor_fails():
    report = check(frame(neutral_share=0.005))
    assert not report.passed
    assert any("rarest class share" in f for f in report.failures)


def test_missing_class_fails():
    df = frame()
    df = df[df["label"] != 1]
    report = check(df)
    assert not report.passed
    assert any("expected 3 classes" in f for f in report.failures)


def test_severe_but_tolerated_imbalance_passes():
    report = check(frame(neutral_share=0.04))
    assert report.passed
    assert report.min_class_share == pytest.approx(0.04, abs=1e-3)
    assert report.n_classes == 3


def test_validate_raises_on_failure():
    with pytest.raises(DataQualityError, match="rows"):
        validate(frame(n=10))


def test_validate_returns_report_on_success():
    assert validate(frame()).passed
