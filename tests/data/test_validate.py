import pandas as pd
import pytest

from sentiment.data.validate import DataQualityError, check, validate


def frame(n: int = 2000, positive_ratio: float = 0.5) -> pd.DataFrame:
    n_pos = int(n * positive_ratio)
    labels = [1] * n_pos + [0] * (n - n_pos)
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


def test_label_imbalance_beyond_tolerance_fails():
    report = check(frame(positive_ratio=0.9))
    assert not report.passed
    assert any("balance" in f for f in report.failures)


def test_validate_raises_on_failure():
    with pytest.raises(DataQualityError, match="rows"):
        validate(frame(n=10))


def test_validate_returns_report_on_success():
    assert validate(frame()).passed
