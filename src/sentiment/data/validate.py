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
    class_shares: dict[int, float]
    min_class_share: float
    n_classes: int
    max_text_length: int
    passed: bool
    failures: tuple[str, ...]


def check(
    df: pd.DataFrame,
    *,
    min_rows: int = 1_000,
    max_empty_ratio: float = 0.0,
    max_duplicate_ratio: float = 0.05,
    min_class_share: float = 0.02,
    expected_classes: int = 3,
) -> QualityReport:
    """Evaluate ``df`` against the quality rules without raising.

    The class-balance rule is a floor on the rarest class rather than a deviation
    from an expected prior. A prior only exists for a balanced binary problem;
    this dataset is three-class with ``neutral`` at roughly 4%, so a
    deviation-from-0.5 rule would fail every healthy run. What actually needs
    catching is a class disappearing — a split with almost no ``neutral`` rows
    cannot train or score that class, and the resulting macro-F1 is meaningless.

    Args:
        df: Normalised frame with ``label`` and ``text``.
        min_rows: Minimum acceptable row count.
        max_empty_ratio: Maximum share of blank texts tolerated.
        max_duplicate_ratio: Maximum share of duplicated texts tolerated.
        min_class_share: Minimum share the rarest class must hold.
        expected_classes: Number of distinct labels the frame must contain.

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
            class_shares={},
            min_class_share=0.0,
            n_classes=0,
            max_text_length=0,
            passed=False,
            failures=(f"missing column(s): {sorted(missing)}",),
        )

    n_rows: int = len(df)
    text: pd.Series = df["text"].astype(str)  # type: ignore[assignment]
    stripped: pd.Series = text.str.strip()  # type: ignore[assignment]
    n_empty = int(stripped.eq("").sum())
    n_duplicates = int(text.duplicated().sum())
    lengths: pd.Series = text.str.len()  # type: ignore[assignment]
    max_len = int(lengths.max()) if n_rows else 0

    counts = df["label"].value_counts().to_dict() if n_rows else {}
    shares = {int(label): count / n_rows for label, count in counts.items()} if n_rows else {}
    rarest = min(shares.values()) if shares else 0.0

    if n_rows < min_rows:
        failures.append(f"too few rows: {n_rows} < {min_rows}")
    if n_rows and n_empty / n_rows > max_empty_ratio:
        failures.append(f"empty text ratio {n_empty / n_rows:.4f} exceeds {max_empty_ratio}")
    if n_rows and n_duplicates / n_rows > max_duplicate_ratio:
        failures.append(
            f"duplicate ratio {n_duplicates / n_rows:.4f} exceeds {max_duplicate_ratio}"
        )
    if n_rows and len(shares) != expected_classes:
        failures.append(f"expected {expected_classes} classes, found {sorted(shares)}")
    if n_rows and rarest < min_class_share:
        failures.append(f"rarest class share {rarest:.4f} is below {min_class_share}")

    return QualityReport(
        n_rows=n_rows,
        n_empty_text=n_empty,
        n_duplicates=n_duplicates,
        class_shares=shares,
        min_class_share=rarest,
        n_classes=len(shares),
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
