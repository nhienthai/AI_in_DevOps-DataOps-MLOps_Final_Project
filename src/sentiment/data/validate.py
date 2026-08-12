"""The data quality gate. Failures stop the pipeline rather than warn."""

from dataclasses import dataclass

import pandas as pd

REQUIRED_COLUMNS = ("label", "text")


class DataQualityError(Exception):
    """Raised when a frame fails the quality gate."""


@dataclass(frozen=True)
class QualityReport:
    """The outcome of a quality check, suitable for logging as MLflow params."""

    n_rows: int
    n_empty_text: int
    n_duplicates: int
    positive_ratio: float
    max_text_length: int
    passed: bool
    failures: tuple[str, ...]


def check(
    df: pd.DataFrame,
    *,
    min_rows: int = 1_000,
    max_empty_ratio: float = 0.0,
    max_duplicate_ratio: float = 0.05,
    balance_tolerance: float = 0.1,
) -> QualityReport:
    """Evaluate ``df`` against the quality rules without raising.

    Args:
        df: Normalised frame with ``label`` and ``text``.
        min_rows: Minimum acceptable row count.
        max_empty_ratio: Maximum share of blank texts tolerated.
        max_duplicate_ratio: Maximum share of duplicated texts tolerated.
        balance_tolerance: Maximum deviation of the positive ratio from 0.5.

    Returns:
        A :class:`QualityReport`; inspect ``passed`` and ``failures``.
    """
    failures: list[str] = []

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        return QualityReport(
            n_rows=len(df),
            n_empty_text=0,
            n_duplicates=0,
            positive_ratio=0.0,
            max_text_length=0,
            passed=False,
            failures=(f"missing column(s): {sorted(missing)}",),
        )

    n_rows: int = len(df)
    text: pd.Series = df["text"].astype(str)  # type: ignore[assignment]
    stripped: pd.Series = text.str.strip()  # type: ignore[assignment]
    n_empty: int = stripped.eq("").sum().item()  # type: ignore[union-attr]
    n_duplicates: int = text.duplicated().sum().item()  # type: ignore[union-attr]
    positive_ratio: float = df["label"].mean() if n_rows else 0.0  # type: ignore[assignment]
    lengths: pd.Series = text.str.len()  # type: ignore[assignment]
    max_len: int = lengths.max().item() if n_rows else 0  # type: ignore[union-attr]

    if n_rows < min_rows:
        failures.append(f"too few rows: {n_rows} < {min_rows}")
    if n_rows and n_empty / n_rows > max_empty_ratio:
        failures.append(f"empty text ratio {n_empty / n_rows:.4f} exceeds {max_empty_ratio}")
    if n_rows and n_duplicates / n_rows > max_duplicate_ratio:
        failures.append(
            f"duplicate ratio {n_duplicates / n_rows:.4f} exceeds {max_duplicate_ratio}"
        )
    if abs(positive_ratio - 0.5) > balance_tolerance:
        failures.append(f"label balance {positive_ratio:.3f} deviates beyond {balance_tolerance}")

    return QualityReport(
        n_rows=n_rows,
        n_empty_text=n_empty,
        n_duplicates=n_duplicates,
        positive_ratio=positive_ratio,
        max_text_length=max_len,
        passed=not failures,
        failures=tuple(failures),
    )


def validate(df: pd.DataFrame, **kwargs: float | int) -> QualityReport:
    """Run :func:`check` and raise :class:`DataQualityError` on any failure."""
    report = check(df, **kwargs)  # type: ignore[arg-type]
    if not report.passed:
        raise DataQualityError("; ".join(report.failures))
    return report
