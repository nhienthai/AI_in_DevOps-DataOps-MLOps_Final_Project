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


VSFC_COLUMNS = ("Sentence", "Encoded_sentiment")


def normalise_vsfc(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a UIT-VSFC frame to the shared ``["label", "text"]`` schema.

    UIT-VSFC ships one sentence per row with a three-class encoded sentiment, so
    there is nothing to concatenate — unlike ``amazon_polarity``, which splits a
    review across ``title`` and ``content``. Both land on the same two columns so
    that everything downstream stays dataset-agnostic.

    Args:
        df: Frame carrying at least ``Sentence`` and ``Encoded_sentiment``.

    Returns:
        A frame with exactly ``["label", "text"]``.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(VSFC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing raw columns: {sorted(missing)}")

    text: pd.Series = df["Sentence"].fillna("").astype(str).str.strip()  # type: ignore[assignment]
    return pd.DataFrame({"label": df["Encoded_sentiment"].astype("int64"), "text": text})


NORMALISERS = {
    "amazon_polarity": normalise,
    "tridm/UIT-VSFC": normalise_vsfc,
}


def normalise_for(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the normaliser registered for ``dataset_name``.

    Raises:
        KeyError: If the dataset has no registered normaliser. Failing loudly
            beats guessing a column layout and silently training on nonsense.
    """
    try:
        normaliser = NORMALISERS[dataset_name]
    except KeyError as exc:
        raise KeyError(
            f"no normaliser registered for '{dataset_name}'; "
            f"known datasets: {sorted(NORMALISERS)}"
        ) from exc
    return normaliser(df)


def ingest(split: str, output_dir: Path, dataset_name: str = "tridm/UIT-VSFC") -> Path:
    """Download one split from HuggingFace and write it as normalised Parquet."""
    from datasets import Dataset, load_dataset

    ds: Dataset = load_dataset(dataset_name, split=split)  # type: ignore[assignment]
    result = ds.to_pandas()
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"Expected DataFrame from to_pandas(), got {type(result)}")
    raw: pd.DataFrame = result
    return write_parquet(normalise_for(dataset_name, raw), output_dir / f"{split}.parquet")
