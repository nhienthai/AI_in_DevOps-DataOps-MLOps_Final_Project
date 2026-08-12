from pathlib import Path

import pandas as pd

from sentiment.data.ingest import normalise, write_parquet

RAW = pd.DataFrame(
    {
        "label": [1, 0],
        "title": ["Great product", "Broke instantly"],
        "content": ["Works exactly as described.", "Snapped on day two."],
    }
)


def test_normalise_produces_only_label_and_text():
    out = normalise(RAW)
    assert list(out.columns) == ["label", "text"]


def test_normalise_joins_title_and_content():
    out = normalise(RAW)
    assert out.loc[0, "text"] == "Great product. Works exactly as described."


def test_normalise_handles_missing_title():
    df = pd.DataFrame({"label": [1], "title": [None], "content": ["Good."]})
    assert normalise(df).loc[0, "text"] == "Good."


def test_normalise_keeps_labels_as_int():
    assert normalise(RAW)["label"].dtype == "int64"


def test_write_parquet_roundtrips(tmp_path: Path):
    path = write_parquet(normalise(RAW), tmp_path / "train.parquet")
    assert path.exists()
    assert len(pd.read_parquet(path)) == 2
