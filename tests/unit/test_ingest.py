from pathlib import Path

import pandas as pd
import pytest

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


def test_vsfc_normalises_to_the_shared_schema():
    from sentiment.data.ingest import normalise_vsfc

    raw = pd.DataFrame(
        {
            "Sentence": ["  thầy dạy hay  ", "bài giảng khó hiểu"],
            "Encoded_sentiment": [2, 0],
            "Topic": ["lecturer", "lecturer"],
        }
    )
    out = normalise_vsfc(raw)

    assert list(out.columns) == ["label", "text"]
    assert out["text"].tolist() == ["thầy dạy hay", "bài giảng khó hiểu"]
    assert out["label"].tolist() == [2, 0]


def test_vsfc_missing_columns_raise():
    from sentiment.data.ingest import normalise_vsfc

    with pytest.raises(ValueError, match="missing raw columns"):
        normalise_vsfc(pd.DataFrame({"Sentence": ["a"]}))


def test_normaliser_registry_dispatches_by_dataset():
    from sentiment.data.ingest import normalise_for

    raw = pd.DataFrame({"Sentence": ["thầy dạy hay"], "Encoded_sentiment": [2]})
    out = normalise_for("tridm/UIT-VSFC", raw)
    assert out["text"].tolist() == ["thầy dạy hay"]


def test_unknown_dataset_fails_loudly():
    from sentiment.data.ingest import normalise_for

    with pytest.raises(KeyError, match="no normaliser registered"):
        normalise_for("some/unknown-dataset", pd.DataFrame({"Sentence": ["a"]}))
