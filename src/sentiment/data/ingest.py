"""Download the raw dataset and normalise it to a two-column frame."""

from pathlib import Path

import pandas as pd

RAW_COLUMNS = ("label", "title", "content")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse ``title`` and ``content`` into a single ``text`` column.

    Args:
        df: Frame carrying at least ``label``, ``title`` and ``content``.

    Returns:
        A frame with exactly ``["label", "text"]``.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")

    title: pd.Series = df["title"].fillna("").astype(str).str.strip()  # type: ignore[assignment]
    content: pd.Series = (  # type: ignore[assignment]
        df["content"].fillna("").astype(str).str.strip()
    )
    combined: pd.Series = (title + ". " + content).astype(str)  # type: ignore[assignment]
    text: pd.Series = (  # type: ignore[assignment]
        combined.str.strip().str.removeprefix(". ").str.strip()
    )

    return pd.DataFrame({"label": df["label"].astype("int64"), "text": text})


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def ingest(split: str, output_dir: Path, dataset_name: str = "amazon_polarity") -> Path:
    """Download one split from HuggingFace and write it as normalised Parquet."""
    from datasets import Dataset, load_dataset

    ds: Dataset = load_dataset(dataset_name, split=split)  # type: ignore[assignment]
    result = ds.to_pandas()
    # When split is a str, to_pandas() always returns DataFrame — guard for type narrowing.
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from to_pandas(), got {type(result)}")
    raw: pd.DataFrame = result
    return write_parquet(normalise(raw), output_dir / f"{split}.parquet")
