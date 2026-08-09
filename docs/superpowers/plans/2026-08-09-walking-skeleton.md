# Walking Skeleton Implementation Plan (Week 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the complete six-service Docker Compose stack, serving predictions from a deterministic stub model, with the data pipeline, Prometheus instrumentation, and green CI all working end-to-end — so that Week 2 only has to swap the stub for a real model.

**Architecture:** A `src/sentiment/` package split into `data/` (ingest, validate, preprocess), `serving/` (FastAPI app, predictor protocol, metrics), and `config.py`. The serving layer depends on a `Predictor` Protocol, not on a concrete model, so Week 2's DistilBERT drops in without touching the API. All ML metrics are computed in-process. Docker Compose runs api, mlflow, postgres, prometheus, grafana, alertmanager with health checks throughout.

**Tech Stack:** Python 3.11, FastAPI, pydantic v2 + pydantic-settings, prometheus-client, pandas + pyarrow, numpy, HuggingFace `datasets`, pytest + httpx, ruff, mypy, Docker Compose, GitHub Actions.

## Global Constraints

- Python 3.11 exactly (pinned in `pyproject.toml`, Dockerfiles, and CI matrix).
- Package root is `src/sentiment/`; all imports are absolute (`from sentiment.data import ...`).
- Every public function has type hints and a docstring; `mypy --strict` must pass on `src/`.
- Coverage gate `--cov-fail-under=80` is active from Task 1 and never lowered.
- `score` is P(positive) in [0,1]; `confidence` is `max(score, 1-score)` in [0.5,1]. Never conflate.
- Labels: `0` = negative, `1` = positive (matches HuggingFace `amazon_polarity`).
- Label strings in API responses: `"positive"` / `"negative"` (lowercase).
- No raw review text in logs — only lengths, hashes, and request ids.
- Settings are read via `sentiment.config.get_settings()`, never `os.environ` directly.
- Env var prefix is `SENTIMENT_`.
- API routes live under `/api/v1`; operational routes (`/health`, `/ready`, `/metrics`) are unprefixed.
- Commit after every task. Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`, `ci:`).

---

### Task 1: Project scaffold, tooling, and configuration

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.env.example`
- Create: `src/sentiment/__init__.py`, `src/sentiment/config.py`
- Create: `src/sentiment/data/__init__.py`, `src/sentiment/serving/__init__.py`
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `sentiment.config.Settings` (pydantic-settings model) and
  `sentiment.config.get_settings() -> Settings` (lru_cached). Every later task
  reads configuration through `get_settings()`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "sentiment-service"
version = "0.1.0"
description = "Real-time sentiment analysis service for e-commerce reviews"
requires-python = "==3.11.*"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "prometheus-client>=0.21",
    "pandas>=2.2",
    "pyarrow>=18.0",
    "numpy>=1.26",
    "datasets>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "pytest-asyncio>=0.24",
    "httpx>=0.27",
    "ruff>=0.7",
    "mypy>=1.13",
    "pandas-stubs",
]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B", "SIM", "D"]
ignore = ["D203", "D213"]

[tool.ruff.lint.per-file-ignores]
"tests/*" = ["D"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src"]

[[tool.mypy.overrides]]
module = ["datasets.*", "sklearn.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
markers = [
    "slow: tests that load or train large models (excluded from PR CI)",
    "integration: tests requiring a running stack",
]
addopts = "--cov=sentiment --cov-report=term-missing --cov-report=xml --cov-fail-under=80"

[tool.coverage.run]
source = ["src/sentiment"]
omit = ["*/__init__.py"]
```

- [ ] **Step 2: Create the package skeleton and Makefile**

```bash
mkdir -p src/sentiment/data src/sentiment/serving tests/unit tests/integration tests/data_quality tests/model
touch src/sentiment/__init__.py src/sentiment/data/__init__.py src/sentiment/serving/__init__.py
```

`Makefile`:

```makefile
.PHONY: install lint typecheck test up down smoke

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy

test:
	pytest -m "not slow" -v

up:
	docker compose up -d

down:
	docker compose down -v

smoke:
	pytest tests/integration -m integration -v
```

- [ ] **Step 3: Write the failing test for configuration**

`tests/unit/test_config.py`:

```python
import pytest

from sentiment.config import Settings, get_settings


def test_defaults_match_spec():
    s = Settings()
    assert s.dataset_name == "amazon_polarity"
    assert s.train_size == 200_000
    assert s.val_size == 25_000
    assert s.test_size == 25_000
    assert s.random_seed == 42
    assert s.max_batch_size == 64
    assert s.max_text_length == 5_000
    assert s.low_confidence_threshold == 0.7
    assert s.drift_window_size == 1_000


def test_env_prefix_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SENTIMENT_MAX_BATCH_SIZE", "8")
    assert Settings().max_batch_size == 8


def test_get_settings_is_cached():
    assert get_settings() is get_settings()
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/unit/test_config.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.config'`

- [ ] **Step 5: Implement `src/sentiment/config.py`**

```python
"""Application configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tunable settings, sourced from ``SENTIMENT_``-prefixed env vars."""

    model_config = SettingsConfigDict(
        env_prefix="SENTIMENT_", env_file=".env", extra="ignore"
    )

    # -- data ---------------------------------------------------------------
    dataset_name: str = "amazon_polarity"
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    train_size: int = 200_000
    val_size: int = 25_000
    test_size: int = 25_000
    random_seed: int = 42

    # -- serving ------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_text_length: int = 5_000
    max_batch_size: int = 64
    low_confidence_threshold: float = 0.7
    drift_window_size: int = 1_000

    # -- mlflow -------------------------------------------------------------
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "sentiment-amazon-polarity"
    model_stage: str = "Production"

    # -- observability ------------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
```

- [ ] **Step 6: Create `.env.example`**

```bash
# Copy to .env and adjust. Never commit .env.
SENTIMENT_LOG_LEVEL=INFO
SENTIMENT_MLFLOW_TRACKING_URI=http://mlflow:5000
SENTIMENT_MAX_BATCH_SIZE=64
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=change-me-locally
POSTGRES_DB=mlflow
GRAFANA_ADMIN_PASSWORD=change-me-locally
```

- [ ] **Step 7: Run tests and lint to verify they pass**

Run: `pip install -e ".[dev]" && pytest tests/unit/test_config.py -v --no-cov && ruff check src tests && mypy`
Expected: 3 passed, no lint errors, no type errors.

- [ ] **Step 8: Generate `requirements.txt`**

The brief lists `requirements.txt` as a required file, but `pyproject.toml` is the
source of truth. Generate it as a pinned lock of the runtime dependencies only,
and add a `deps` target so it is regenerated rather than hand-edited.

Append to the `Makefile`:

```makefile
.PHONY: deps

deps:
	pip install pip-tools
	pip-compile --output-file=requirements.txt --strip-extras pyproject.toml
```

Run: `make deps`
Expected: `requirements.txt` exists and pins every runtime dependency with hashes
of the resolved versions. Re-run this target whenever `pyproject.toml` changes.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.txt Makefile .env.example src tests
git commit -m "chore: scaffold package, tooling, and settings"
```

---

### Task 2: Raw data ingestion

**Files:**
- Create: `src/sentiment/data/ingest.py`
- Test: `tests/unit/test_ingest.py`

**Interfaces:**
- Consumes: `sentiment.config.get_settings`.
- Produces:
  - `normalise(df: pd.DataFrame) -> pd.DataFrame` — takes raw columns
    `label, title, content`, returns exactly `["label", "text"]` where
    `text = f"{title}. {content}"` stripped, and `label` is `int64`.
  - `write_parquet(df: pd.DataFrame, path: Path) -> Path`
  - `ingest(split: str, output_dir: Path, dataset_name: str) -> Path` — downloads
    from HuggingFace and writes `{output_dir}/{split}.parquet`. Integration-marked.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_ingest.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_ingest.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.data.ingest'`

- [ ] **Step 3: Implement `src/sentiment/data/ingest.py`**

```python
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

    title = df["title"].fillna("").astype(str).str.strip()
    content = df["content"].fillna("").astype(str).str.strip()
    text = (title + ". " + content).str.strip().str.removeprefix(". ").str.strip()

    return pd.DataFrame({"label": df["label"].astype("int64"), "text": text})


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write ``df`` to ``path`` as Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def ingest(split: str, output_dir: Path, dataset_name: str = "amazon_polarity") -> Path:
    """Download one split from HuggingFace and write it as normalised Parquet."""
    from datasets import load_dataset

    raw = load_dataset(dataset_name, split=split).to_pandas()
    return write_parquet(normalise(raw), output_dir / f"{split}.parquet")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_ingest.py -v --no-cov`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/data/ingest.py tests/unit/test_ingest.py
git commit -m "feat: add raw data ingestion and normalisation"
```

---

### Task 3: Data quality gate

**Files:**
- Create: `src/sentiment/data/validate.py`
- Test: `tests/data_quality/test_validate.py`

**Interfaces:**
- Consumes: `normalise` output shape (`["label", "text"]`).
- Produces:
  - `DataQualityError(Exception)`
  - `QualityReport` — frozen dataclass with fields `n_rows: int`,
    `n_empty_text: int`, `n_duplicates: int`, `positive_ratio: float`,
    `max_text_length: int`, `passed: bool`, `failures: tuple[str, ...]`.
  - `check(df, *, min_rows=1000, max_empty_ratio=0.0, max_duplicate_ratio=0.05, balance_tolerance=0.1) -> QualityReport`
  - `validate(df, **kwargs) -> QualityReport` — same signature, raises
    `DataQualityError` when `report.passed` is False.

- [ ] **Step 1: Write the failing tests**

`tests/data_quality/test_validate.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/data_quality/test_validate.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.data.validate'`

- [ ] **Step 3: Implement `src/sentiment/data/validate.py`**

```python
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

    n_rows = len(df)
    text = df["text"].astype(str)
    n_empty = int(text.str.strip().eq("").sum())
    n_duplicates = int(text.duplicated().sum())
    positive_ratio = float(df["label"].mean()) if n_rows else 0.0
    max_len = int(text.str.len().max()) if n_rows else 0

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/data_quality/test_validate.py -v --no-cov`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/data/validate.py tests/data_quality/test_validate.py
git commit -m "feat: add data quality gate with failing validation"
```

---

### Task 4: Splits and the drift reference

**Files:**
- Create: `src/sentiment/data/preprocess.py`
- Test: `tests/unit/test_preprocess.py`

**Interfaces:**
- Consumes: validated frames with `["label", "text"]`.
- Produces:
  - `DriftReference` — frozen dataclass with `length_bin_edges: tuple[float, ...]`,
    `length_bin_freqs: tuple[float, ...]` (length is `len(edges) - 1`, sums to 1.0),
    `positive_prior: float`; methods `to_dict() -> dict` and classmethod
    `from_dict(dict) -> DriftReference`.
  - `stratified_subsample(df, n, seed) -> pd.DataFrame`
  - `make_splits(train_df, test_df, *, train_size, val_size, test_size, seed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
    returning `(train, val, test)`.
  - `build_drift_reference(texts, labels, n_bins=10) -> DriftReference`

  Task 6 consumes `DriftReference`; Week 2 logs `to_dict()` as an MLflow artifact.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_preprocess.py`:

```python
import pandas as pd

from sentiment.data.preprocess import (
    DriftReference,
    build_drift_reference,
    make_splits,
    stratified_subsample,
)


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"label": [i % 2 for i in range(n)], "text": [f"review {i}" for i in range(n)]}
    )


def test_subsample_returns_requested_size():
    assert len(stratified_subsample(frame(1000), 100, seed=42)) == 100


def test_subsample_preserves_label_balance():
    out = stratified_subsample(frame(1000), 100, seed=42)
    assert abs(out["label"].mean() - 0.5) < 0.05


def test_subsample_is_deterministic_for_a_seed():
    a = stratified_subsample(frame(1000), 100, seed=42)
    b = stratified_subsample(frame(1000), 100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_subsample_larger_than_source_returns_all():
    assert len(stratified_subsample(frame(50), 500, seed=42)) == 50


def test_make_splits_sizes_and_disjointness():
    train, val, test = make_splits(
        frame(1000), frame(400), train_size=200, val_size=50, test_size=40, seed=42
    )
    assert (len(train), len(val), len(test)) == (200, 50, 40)
    assert set(train["text"]).isdisjoint(set(val["text"]))


def test_drift_reference_frequencies_sum_to_one():
    ref = build_drift_reference([("x" * (i % 50 + 1)) for i in range(500)], [i % 2 for i in range(500)])
    assert abs(sum(ref.length_bin_freqs) - 1.0) < 1e-9
    assert len(ref.length_bin_freqs) == len(ref.length_bin_edges) - 1


def test_drift_reference_records_positive_prior():
    ref = build_drift_reference(["a", "bb", "ccc", "dddd"], [1, 1, 0, 0])
    assert ref.positive_prior == 0.5


def test_drift_reference_roundtrips_through_dict():
    ref = build_drift_reference([("x" * (i % 30 + 1)) for i in range(200)], [i % 2 for i in range(200)])
    assert DriftReference.from_dict(ref.to_dict()) == ref
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_preprocess.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.data.preprocess'`

- [ ] **Step 3: Implement `src/sentiment/data/preprocess.py`**

```python
"""Deterministic splitting and the drift reference distribution.

The drift reference is built here, beside the splits, so that it always
describes the data the model was actually trained on.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftReference:
    """The training-time input distribution, logged alongside the model."""

    length_bin_edges: tuple[float, ...]
    length_bin_freqs: tuple[float, ...]
    positive_prior: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "length_bin_edges": list(self.length_bin_edges),
            "length_bin_freqs": list(self.length_bin_freqs),
            "positive_prior": self.positive_prior,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftReference":
        """Rebuild a reference from :meth:`to_dict` output."""
        return cls(
            length_bin_edges=tuple(float(x) for x in payload["length_bin_edges"]),
            length_bin_freqs=tuple(float(x) for x in payload["length_bin_freqs"]),
            positive_prior=float(payload["positive_prior"]),
        )


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Take ``n`` rows preserving the label distribution, reproducibly."""
    if n >= len(df):
        return df.reset_index(drop=True)
    fraction = n / len(df)
    sampled = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=max(1, round(len(g) * fraction)), random_state=seed))
        .sample(frac=1.0, random_state=seed)
        .head(n)
        .reset_index(drop=True)
    )
    return sampled


def make_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carve train/validation from the train frame and test from the test frame.

    Validation is drawn from the training source so the test split stays
    untouched and comparable across experiments.
    """
    pool = stratified_subsample(train_df, train_size + val_size, seed=seed)
    train = pool.iloc[:train_size].reset_index(drop=True)
    val = pool.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test = stratified_subsample(test_df, test_size, seed=seed)
    return train, val, test


def build_drift_reference(
    texts: Sequence[str], labels: Sequence[int], n_bins: int = 10
) -> DriftReference:
    """Summarise the training input distribution as quantile length bins."""
    lengths = np.array([len(t) for t in texts], dtype=float)
    edges = np.unique(np.quantile(lengths, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([lengths.min(), lengths.min() + 1.0])
    edges[0] = 0.0
    edges[-1] = float(lengths.max()) * 10.0

    counts, _ = np.histogram(lengths, bins=edges)
    freqs = counts / counts.sum()

    return DriftReference(
        length_bin_edges=tuple(float(e) for e in edges),
        length_bin_freqs=tuple(float(f) for f in freqs),
        positive_prior=float(np.mean(labels)),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_preprocess.py -v --no-cov`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/data/preprocess.py tests/unit/test_preprocess.py
git commit -m "feat: add deterministic splits and drift reference"
```

---

### Task 5: Predictor protocol and stub model

**Files:**
- Create: `src/sentiment/serving/predictor.py`
- Test: `tests/unit/test_predictor.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Prediction` — frozen dataclass: `label: str` (`"positive"`/`"negative"`),
    `score: float` (P(positive)), `confidence: float` (`max(score, 1-score)`),
    `truncated: bool`.
  - `Predictor` — `typing.Protocol` with attribute `version: str` and method
    `predict(self, texts: Sequence[str]) -> list[Prediction]`.
  - `StubPredictor(max_chars: int = 5000)` implementing `Predictor` with
    `version = "stub-0"`. Week 2 replaces it with `TransformerPredictor`
    satisfying the same Protocol — the API must not change.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_predictor.py`:

```python
from sentiment.serving.predictor import Prediction, StubPredictor


def test_version_is_reported():
    assert StubPredictor().version == "stub-0"


def test_returns_one_prediction_per_text():
    assert len(StubPredictor().predict(["a", "b", "c"])) == 3


def test_prediction_is_deterministic_for_the_same_text():
    p = StubPredictor()
    assert p.predict(["hello world"])[0] == p.predict(["hello world"])[0]


def test_label_agrees_with_score():
    for pred in StubPredictor().predict([f"review {i}" for i in range(50)]):
        assert pred.label == ("positive" if pred.score >= 0.5 else "negative")


def test_confidence_is_distance_from_the_decision_boundary():
    for pred in StubPredictor().predict([f"review {i}" for i in range(50)]):
        assert pred.confidence == max(pred.score, 1.0 - pred.score)
        assert 0.5 <= pred.confidence <= 1.0


def test_long_text_is_flagged_as_truncated():
    pred = StubPredictor(max_chars=10).predict(["x" * 50])[0]
    assert pred.truncated is True


def test_short_text_is_not_flagged():
    assert StubPredictor(max_chars=10).predict(["short"])[0].truncated is False


def test_prediction_is_immutable():
    pred = Prediction(label="positive", score=0.9, confidence=0.9, truncated=False)
    try:
        pred.score = 0.1  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Prediction should be frozen")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_predictor.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.serving.predictor'`

- [ ] **Step 3: Implement `src/sentiment/serving/predictor.py`**

```python
"""The prediction interface and a deterministic stand-in model.

``StubPredictor`` exists so the whole stack can be integrated before a real
model is trained. It is replaced in Week 2 by a class satisfying the same
:class:`Predictor` protocol; nothing in the API layer changes.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Prediction:
    """A single sentiment prediction.

    Attributes:
        label: ``"positive"`` or ``"negative"``.
        score: Probability that the text is positive, in ``[0, 1]``.
        confidence: ``max(score, 1 - score)``, in ``[0.5, 1]``.
        truncated: Whether the input exceeded the model's window.
    """

    label: str
    score: float
    confidence: float
    truncated: bool


@runtime_checkable
class Predictor(Protocol):
    """Anything the serving layer can predict with."""

    version: str

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Score a batch of texts, returning one prediction per input."""
        ...


def _to_prediction(score: float, truncated: bool) -> Prediction:
    """Build a :class:`Prediction`, deriving label and confidence from ``score``."""
    return Prediction(
        label="positive" if score >= 0.5 else "negative",
        score=score,
        confidence=max(score, 1.0 - score),
        truncated=truncated,
    )


class StubPredictor:
    """Deterministic placeholder: the score is a hash of the text.

    Deterministic rather than random so that tests, dashboards, and demos are
    reproducible before a real model exists.
    """

    version = "stub-0"

    def __init__(self, max_chars: int = 5_000) -> None:
        self._max_chars = max_chars

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Score each text by hashing it into ``[0, 1]``."""
        predictions: list[Prediction] = []
        for text in texts:
            truncated = len(text) > self._max_chars
            digest = hashlib.sha256(text[: self._max_chars].encode("utf-8")).digest()
            score = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
            predictions.append(_to_prediction(score, truncated))
        return predictions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_predictor.py -v --no-cov`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/serving/predictor.py tests/unit/test_predictor.py
git commit -m "feat: add predictor protocol and deterministic stub"
```

---

### Task 6: Prometheus metrics and drift tracking

**Files:**
- Create: `src/sentiment/serving/metrics.py`
- Test: `tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: `sentiment.data.preprocess.DriftReference`.
- Produces:
  - `population_stability_index(expected, actual, eps=1e-6) -> float`
  - `DriftTracker(reference: DriftReference, window_size: int = 1000)` with
    `observe(text_length: int) -> None`, `psi() -> float`, and `__len__`.
    Returns `0.0` until at least `MIN_OBSERVATIONS` (30) samples.
  - Module-level collectors: `REQUESTS`, `DURATION`, `PREDICTIONS`, `CONFIDENCE`,
    `LOW_CONFIDENCE`, `INPUT_LENGTH`, `DRIFT_PSI`, `MODEL_INFO`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_metrics.py`:

```python
from sentiment.data.preprocess import DriftReference
from sentiment.serving.metrics import DriftTracker, population_stability_index

REF = DriftReference(
    length_bin_edges=(0.0, 10.0, 20.0, 30.0, 1000.0),
    length_bin_freqs=(0.25, 0.25, 0.25, 0.25),
    positive_prior=0.5,
)


def test_psi_is_zero_for_identical_distributions():
    assert population_stability_index([0.25] * 4, [0.25] * 4) < 1e-9


def test_psi_grows_with_divergence():
    mild = population_stability_index([0.25] * 4, [0.30, 0.25, 0.25, 0.20])
    severe = population_stability_index([0.25] * 4, [0.85, 0.05, 0.05, 0.05])
    assert 0 < mild < severe


def test_psi_flags_significant_shift_above_conventional_threshold():
    assert population_stability_index([0.25] * 4, [0.85, 0.05, 0.05, 0.05]) > 0.2


def test_psi_tolerates_empty_bins():
    assert population_stability_index([0.25] * 4, [1.0, 0.0, 0.0, 0.0]) > 0.0


def test_tracker_reports_zero_before_enough_observations():
    tracker = DriftTracker(REF, window_size=100)
    for _ in range(5):
        tracker.observe(5)
    assert tracker.psi() == 0.0


def test_tracker_reports_low_psi_on_matching_traffic():
    tracker = DriftTracker(REF, window_size=1000)
    for length in [5, 15, 25, 100] * 50:
        tracker.observe(length)
    assert tracker.psi() < 0.1


def test_tracker_reports_high_psi_on_shifted_traffic():
    tracker = DriftTracker(REF, window_size=1000)
    for _ in range(200):
        tracker.observe(5)
    assert tracker.psi() > 0.2


def test_tracker_window_is_bounded():
    tracker = DriftTracker(REF, window_size=50)
    for i in range(500):
        tracker.observe(i)
    assert len(tracker) == 50


def test_tracker_clips_lengths_beyond_the_reference_range():
    tracker = DriftTracker(REF, window_size=100)
    for _ in range(50):
        tracker.observe(10_000_000)
    assert tracker.psi() > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_metrics.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.serving.metrics'`

- [ ] **Step 3: Implement `src/sentiment/serving/metrics.py`**

```python
"""Prometheus collectors and in-process drift tracking.

Drift is computed here rather than in a sidecar because it is a function of the
prediction the API just made; shipping predictions elsewhere to measure them
would add a service without adding information.
"""

from collections import deque
from collections.abc import Sequence

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from sentiment.data.preprocess import DriftReference

MIN_OBSERVATIONS = 30

REQUESTS = Counter(
    "sentiment_requests_total",
    "Total API requests.",
    ["endpoint", "status", "model_version"],
)
DURATION = Histogram(
    "sentiment_request_duration_seconds",
    "Request latency in seconds.",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.5, 5.0),
)
PREDICTIONS = Counter(
    "sentiment_predictions_total",
    "Predictions issued, by label.",
    ["label", "model_version"],
)
CONFIDENCE = Histogram(
    "sentiment_confidence",
    "Distribution of prediction confidence.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
LOW_CONFIDENCE = Counter(
    "sentiment_low_confidence_total",
    "Predictions below the low-confidence threshold.",
)
INPUT_LENGTH = Histogram(
    "sentiment_input_length_chars",
    "Distribution of input text length in characters.",
    buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
)
DRIFT_PSI = Gauge(
    "sentiment_drift_psi",
    "Population stability index of live input lengths against the training reference.",
)
MODEL_INFO = Gauge(
    "sentiment_model_info",
    "Always 1; labels carry the served model's provenance.",
    ["version"],
)


def population_stability_index(
    expected: Sequence[float], actual: Sequence[float], eps: float = 1e-6
) -> float:
    """Return the PSI between two binned distributions.

    Conventional reading: below 0.1 is stable, 0.1-0.2 is a moderate shift, and
    above 0.2 is a significant shift. The alert threshold uses 0.2.

    Args:
        expected: Reference bin frequencies or counts.
        actual: Observed bin frequencies or counts, same length as ``expected``.
        eps: Floor applied to both, so empty bins do not produce infinities.

    Returns:
        The PSI as a non-negative float.
    """
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


class DriftTracker:
    """A bounded rolling window of input lengths, scored against a reference."""

    def __init__(self, reference: DriftReference, window_size: int = 1_000) -> None:
        self._edges = np.asarray(reference.length_bin_edges, dtype=float)
        self._expected = np.asarray(reference.length_bin_freqs, dtype=float)
        self._window: deque[int] = deque(maxlen=window_size)

    def __len__(self) -> int:
        """Return the number of observations currently in the window."""
        return len(self._window)

    def observe(self, text_length: int) -> None:
        """Record one observed input length."""
        self._window.append(text_length)

    def psi(self) -> float:
        """Return the current PSI, or 0.0 before the window is warm."""
        if len(self._window) < MIN_OBSERVATIONS:
            return 0.0
        values = np.clip(
            np.asarray(self._window, dtype=float), self._edges[0], self._edges[-1]
        )
        counts, _ = np.histogram(values, bins=self._edges)
        return population_stability_index(self._expected, counts)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_metrics.py -v --no-cov`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/serving/metrics.py tests/unit/test_metrics.py
git commit -m "feat: add prometheus collectors and PSI drift tracker"
```

---

### Task 7: API schemas, typed errors, and the FastAPI app

**Files:**
- Create: `src/sentiment/serving/schemas.py`, `src/sentiment/serving/errors.py`, `src/sentiment/serving/app.py`
- Test: `tests/integration/test_app_health.py`

**Interfaces:**
- Consumes: `StubPredictor`, `DriftTracker`, `DriftReference`, collectors from Task 6, `get_settings`.
- Produces:
  - Schemas `PredictRequest`, `PredictResponse`, `BatchRequest`, `BatchItem`,
    `BatchResponse`, `ModelInfo`, `ErrorResponse`.
  - `APIError(status_code, error_code, message)` and `install_error_handlers(app)`.
  - `create_app() -> FastAPI` — the app factory used by tests and by uvicorn
    (`sentiment.serving.app:create_app`, with `--factory`).
  - `app.state.predictor`, `app.state.drift` — set during lifespan startup;
    Task 8 reads both.

- [ ] **Step 1: Write the failing tests**

`tests/integration/test_app_health.py`:

```python
import httpx
import pytest

from sentiment.serving.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_health_is_up_and_independent_of_the_model(client: httpx.AsyncClient):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_ready_reports_the_loaded_model(client: httpx.AsyncClient):
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == "stub-0"


async def test_metrics_endpoint_exposes_prometheus_text(client: httpx.AsyncClient):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "sentiment_requests_total" in response.text


async def test_openapi_spec_is_generated(client: httpx.AsyncClient):
    spec = (await client.get("/openapi.json")).json()
    assert spec["info"]["title"] == "Sentiment Service"
    assert "/ready" in spec["paths"]


async def test_unknown_route_returns_typed_error(client: httpx.AsyncClient):
    response = await client.get("/api/v1/nope")
    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "not_found"
    assert "request_id" in body


async def test_every_response_carries_a_request_id_header(client: httpx.AsyncClient):
    assert (await client.get("/health")).headers["x-request-id"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/integration/test_app_health.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.serving.app'`

- [ ] **Step 3: Implement `src/sentiment/serving/schemas.py`**

```python
"""Request and response contracts. The examples here become the OpenAPI docs."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """A single text to score."""

    text: str = Field(
        min_length=1,
        description="Review text to classify.",
        examples=["Arrived quickly and works perfectly."],
    )


class PredictResponse(BaseModel):
    """The result of scoring one text."""

    label: str = Field(description='Either "positive" or "negative".')
    score: float = Field(description="Probability that the text is positive, in [0, 1].")
    confidence: float = Field(description="max(score, 1 - score), in [0.5, 1].")
    model_version: str = Field(description="Version of the model that produced this.")
    truncated: bool = Field(description="Whether the input exceeded the model window.")
    latency_ms: float = Field(description="Server-side processing time in milliseconds.")


class BatchRequest(BaseModel):
    """Several texts to score in one call."""

    texts: list[str] = Field(min_length=1, description="Texts to classify.")


class BatchItem(BaseModel):
    """One element of a batch result: either a prediction or an error."""

    index: int
    prediction: PredictResponse | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Per-item results; individual failures do not fail the request."""

    results: list[BatchItem]


class ModelInfo(BaseModel):
    """Provenance of the model currently being served."""

    model_version: str
    stage: str
    predictor_class: str


class ErrorResponse(BaseModel):
    """The uniform error body returned by every failing endpoint."""

    error_code: str
    message: str
    request_id: str
```

- [ ] **Step 4: Implement `src/sentiment/serving/errors.py`**

```python
"""Typed errors and the handlers that render them uniformly."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    """An error with an explicit status code and machine-readable code."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


_STATUS_TO_CODE = {400: "bad_request", 404: "not_found", 405: "method_not_allowed"}


def _body(request: Request, error_code: str, message: str) -> dict[str, str]:
    """Build the uniform error payload, including the current request id."""
    return {
        "error_code": error_code,
        "message": message,
        "request_id": getattr(request.state, "request_id", "unknown"),
    }


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers so every failure returns an :class:`ErrorResponse` shape."""

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(request, exc.error_code, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body(request, "validation_error", str(exc.errors())),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        code = _STATUS_TO_CODE.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code, content=_body(request, code, str(exc.detail))
        )
```

- [ ] **Step 5: Implement `src/sentiment/serving/app.py` with health, ready, and metrics**

```python
"""The FastAPI application factory."""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentiment.config import get_settings
from sentiment.data.preprocess import DriftReference
from sentiment.serving.errors import install_error_handlers
from sentiment.serving.metrics import MODEL_INFO, DriftTracker
from sentiment.serving.predictor import StubPredictor

# Placeholder reference used while the stub model is in place. Week 2 replaces
# this with the reference logged beside the real model in MLflow.
_BOOTSTRAP_REFERENCE = DriftReference(
    length_bin_edges=(0.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 100_000.0),
    length_bin_freqs=(0.1, 0.2, 0.3, 0.2, 0.15, 0.05),
    positive_prior=0.5,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and drift reference before the app accepts traffic."""
    settings = get_settings()
    app.state.predictor = StubPredictor(max_chars=settings.max_text_length)
    app.state.drift = DriftTracker(
        _BOOTSTRAP_REFERENCE, window_size=settings.drift_window_size
    )
    MODEL_INFO.labels(version=app.state.predictor.version).set(1)
    yield
    app.state.predictor = None


def create_app() -> FastAPI:
    """Build the application. Used by tests and by uvicorn --factory."""
    app = FastAPI(
        title="Sentiment Service",
        version="0.1.0",
        description="Real-time sentiment analysis for e-commerce reviews.",
        lifespan=_lifespan,
    )
    install_error_handlers(app)

    @app.middleware("http")
    async def _request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request.state.request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["x-request-id"] = request.state.request_id
        return response

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        """Liveness: the process is up. Deliberately independent of the model."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    async def ready(request: Request) -> dict[str, str]:
        """Readiness: a model is loaded and the service can serve traffic."""
        predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            from sentiment.serving.errors import APIError

            raise APIError(503, "model_not_ready", "No model is loaded.")
        return {"status": "ready", "model_version": predictor.version}

    @app.get("/metrics", tags=["ops"])
    async def metrics() -> Response:
        """Prometheus exposition endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/integration/test_app_health.py -v --no-cov`
Expected: 6 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sentiment/serving/schemas.py src/sentiment/serving/errors.py src/sentiment/serving/app.py tests/integration/test_app_health.py
git commit -m "feat: add FastAPI app with health, ready, metrics and typed errors"
```

---

### Task 8: Prediction endpoints with instrumentation

**Files:**
- Modify: `src/sentiment/serving/app.py` (add the `/api/v1` router inside `create_app`)
- Test: `tests/integration/test_predict.py`

**Interfaces:**
- Consumes: everything from Tasks 5-7.
- Produces: `POST /api/v1/predict`, `POST /api/v1/predict/batch`,
  `GET /api/v1/model/info`. Week 2 changes only the predictor bound in
  `_lifespan`; these handlers stay as written.

- [ ] **Step 1: Write the failing tests**

`tests/integration/test_predict.py`:

```python
import httpx
import pytest

from sentiment.serving.app import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


async def test_predict_returns_the_full_contract(client: httpx.AsyncClient):
    body = (await client.post("/api/v1/predict", json={"text": "Great product"})).json()
    assert set(body) == {
        "label", "score", "confidence", "model_version", "truncated", "latency_ms"
    }


async def test_confidence_is_derived_from_score(client: httpx.AsyncClient):
    body = (await client.post("/api/v1/predict", json={"text": "Great product"})).json()
    assert body["confidence"] == pytest.approx(max(body["score"], 1 - body["score"]))


async def test_blank_text_is_rejected(client: httpx.AsyncClient):
    response = await client.post("/api/v1/predict", json={"text": "   "})
    assert response.status_code == 422
    assert response.json()["error_code"] in {"validation_error", "empty_text"}


async def test_missing_field_is_rejected(client: httpx.AsyncClient):
    assert (await client.post("/api/v1/predict", json={})).status_code == 422


async def test_oversized_text_returns_413(client: httpx.AsyncClient):
    response = await client.post("/api/v1/predict", json={"text": "x" * 100_000})
    assert response.status_code == 413
    assert response.json()["error_code"] == "text_too_long"


async def test_batch_returns_one_result_per_input(client: httpx.AsyncClient):
    body = (
        await client.post("/api/v1/predict/batch", json={"texts": ["a", "b", "c"]})
    ).json()
    assert [r["index"] for r in body["results"]] == [0, 1, 2]
    assert all(r["error"] is None for r in body["results"])


async def test_batch_isolates_per_item_failures(client: httpx.AsyncClient):
    body = (
        await client.post("/api/v1/predict/batch", json={"texts": ["ok", "  "]})
    ).json()
    assert body["results"][0]["prediction"] is not None
    assert body["results"][1]["error"] is not None


async def test_batch_over_the_limit_is_rejected(client: httpx.AsyncClient):
    response = await client.post(
        "/api/v1/predict/batch", json={"texts": ["x"] * 65}
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "batch_too_large"


async def test_model_info_reports_provenance(client: httpx.AsyncClient):
    body = (await client.get("/api/v1/model/info")).json()
    assert body["model_version"] == "stub-0"
    assert body["predictor_class"] == "StubPredictor"


async def test_predictions_are_counted_in_metrics(client: httpx.AsyncClient):
    await client.post("/api/v1/predict", json={"text": "counted"})
    text = (await client.get("/metrics")).text
    assert "sentiment_predictions_total" in text
    assert 'model_version="stub-0"' in text


async def test_input_length_is_observed(client: httpx.AsyncClient):
    await client.post("/api/v1/predict", json={"text": "measured"})
    assert "sentiment_input_length_chars_count" in (await client.get("/metrics")).text


async def test_inference_routes_appear_in_the_openapi_spec(client: httpx.AsyncClient):
    paths = (await client.get("/openapi.json")).json()["paths"]
    assert {"/api/v1/predict", "/api/v1/predict/batch", "/api/v1/model/info"} <= set(paths)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/integration/test_predict.py -v --no-cov`
Expected: FAIL — every prediction call returns 404.

- [ ] **Step 3: Add the router to `create_app` in `src/sentiment/serving/app.py`**

Insert these imports at the top of the module:

```python
import time

from fastapi import APIRouter

from sentiment.serving.errors import APIError
from sentiment.serving.metrics import (
    CONFIDENCE,
    DRIFT_PSI,
    DURATION,
    INPUT_LENGTH,
    LOW_CONFIDENCE,
    PREDICTIONS,
    REQUESTS,
)
from sentiment.serving.predictor import Prediction
from sentiment.serving.schemas import (
    BatchItem,
    BatchRequest,
    BatchResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)
```

Insert this block inside `create_app`, immediately before `return app`:

```python
    settings = get_settings()
    router = APIRouter(prefix="/api/v1", tags=["inference"])

    def _score_one(request: Request, text: str) -> tuple[Prediction, float]:
        """Validate, score, and instrument a single text."""
        if not text.strip():
            raise APIError(422, "empty_text", "Text must not be blank.")
        if len(text) > settings.max_text_length:
            raise APIError(
                413,
                "text_too_long",
                f"Text exceeds {settings.max_text_length} characters.",
            )

        started = time.perf_counter()
        prediction = request.app.state.predictor.predict([text])[0]
        elapsed = time.perf_counter() - started

        version = request.app.state.predictor.version
        PREDICTIONS.labels(label=prediction.label, model_version=version).inc()
        CONFIDENCE.observe(prediction.confidence)
        INPUT_LENGTH.observe(len(text))
        if prediction.confidence < settings.low_confidence_threshold:
            LOW_CONFIDENCE.inc()

        request.app.state.drift.observe(len(text))
        DRIFT_PSI.set(request.app.state.drift.psi())

        return prediction, elapsed * 1_000.0

    def _to_response(request: Request, prediction: Prediction, latency_ms: float) -> PredictResponse:
        """Render a prediction as the wire contract."""
        return PredictResponse(
            label=prediction.label,
            score=prediction.score,
            confidence=prediction.confidence,
            model_version=request.app.state.predictor.version,
            truncated=prediction.truncated,
            latency_ms=round(latency_ms, 3),
        )

    @router.post("/predict", response_model=PredictResponse)
    async def predict(request: Request, payload: PredictRequest) -> PredictResponse:
        """Classify a single review text."""
        version = request.app.state.predictor.version
        with DURATION.labels(endpoint="predict").time():
            prediction, latency_ms = _score_one(request, payload.text)
        REQUESTS.labels(endpoint="predict", status="200", model_version=version).inc()
        return _to_response(request, prediction, latency_ms)

    @router.post("/predict/batch", response_model=BatchResponse)
    async def predict_batch(request: Request, payload: BatchRequest) -> BatchResponse:
        """Classify several texts, isolating per-item failures."""
        if len(payload.texts) > settings.max_batch_size:
            raise APIError(
                413,
                "batch_too_large",
                f"Batch exceeds {settings.max_batch_size} items.",
            )

        version = request.app.state.predictor.version
        results: list[BatchItem] = []
        with DURATION.labels(endpoint="predict_batch").time():
            for index, text in enumerate(payload.texts):
                try:
                    prediction, latency_ms = _score_one(request, text)
                except APIError as exc:
                    results.append(BatchItem(index=index, error=exc.message))
                else:
                    results.append(
                        BatchItem(
                            index=index,
                            prediction=_to_response(request, prediction, latency_ms),
                        )
                    )
        REQUESTS.labels(
            endpoint="predict_batch", status="200", model_version=version
        ).inc()
        return BatchResponse(results=results)

    @router.get("/model/info", response_model=ModelInfo)
    async def model_info(request: Request) -> ModelInfo:
        """Report which model is currently serving."""
        predictor = request.app.state.predictor
        return ModelInfo(
            model_version=predictor.version,
            stage=settings.model_stage,
            predictor_class=type(predictor).__name__,
        )

    app.include_router(router)
```

- [ ] **Step 4: Run the full test suite to verify it passes**

Run: `pytest tests/ -v --no-cov && ruff check src tests && mypy`
Expected: all tests pass, including the previously-xfailed OpenAPI assertion in
Task 7 (remove the `xfail` marker if you added one).

- [ ] **Step 5: Verify coverage clears the gate**

Run: `pytest -m "not slow"`
Expected: coverage ≥ 80%, no `--cov-fail-under` failure.

- [ ] **Step 6: Commit**

```bash
git add src/sentiment/serving/app.py tests/integration/test_predict.py
git commit -m "feat: add predict, batch and model-info endpoints with metrics"
```

---

### Task 9: Multi-stage container image

**Files:**
- Create: `docker/api.Dockerfile`, `.dockerignore`
- Test: manual build and run (verified again by the Compose smoke test in Task 10)

**Interfaces:**
- Consumes: `pyproject.toml`, `src/`.
- Produces: an image exposing port 8000, running as UID 1000, with a
  `HEALTHCHECK` on `/ready`.

- [ ] **Step 1: Create `.dockerignore`**

```
.git
.github
.venv
data
docs
mlruns
notebooks
tests
**/__pycache__
*.md
.env
.pytest_cache
.mypy_cache
.ruff_cache
htmlcov
```

- [ ] **Step 2: Create `docker/api.Dockerfile`**

```dockerfile
# ---- builder -----------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# ---- runtime -----------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 app \
 && useradd --uid 1000 --gid app --create-home app \
 && apt-get update \
 && apt-get install --no-install-recommends -y curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "sentiment.serving.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Build the image**

Run: `docker build -f docker/api.Dockerfile -t sentiment-api:dev .`
Expected: build succeeds.

- [ ] **Step 4: Verify it runs, is healthy, and is non-root**

```bash
docker run -d --name sentiment-check -p 8000:8000 sentiment-api:dev
sleep 25
curl -fsS http://localhost:8000/ready
docker inspect --format '{{.State.Health.Status}}' sentiment-check
docker exec sentiment-check id -u
docker rm -f sentiment-check
```

Expected: `/ready` returns `{"status":"ready","model_version":"stub-0"}`,
health status is `healthy`, and `id -u` prints `1000`.

- [ ] **Step 5: Commit**

```bash
git add docker/api.Dockerfile .dockerignore
git commit -m "feat: add multi-stage non-root API image"
```

---

### Task 10: Compose stack, monitoring configuration, and smoke test

**Files:**
- Create: `docker-compose.yml`
- Create: `monitoring/prometheus/prometheus.yml`, `monitoring/prometheus/alerts.yml`
- Create: `monitoring/alertmanager/alertmanager.yml`
- Create: `monitoring/grafana/provisioning/datasources/prometheus.yml`
- Create: `monitoring/grafana/provisioning/dashboards/dashboards.yml`
- Test: `tests/integration/test_stack_smoke.py`

**Interfaces:**
- Consumes: the API image from Task 9, the metric names from Task 6.
- Produces: a six-service stack. Week 3 adds dashboard JSON under
  `monitoring/grafana/dashboards/`, which the provisioning file already points at.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
name: sentiment-service

services:
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    ports: ["8000:8000"]
    environment:
      SENTIMENT_MLFLOW_TRACKING_URI: http://mlflow:5000
      SENTIMENT_LOG_LEVEL: ${SENTIMENT_LOG_LEVEL:-INFO}
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/ready"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 20s
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-mlflow}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-mlflow}
      POSTGRES_DB: ${POSTGRES_DB:-mlflow}
    volumes: ["postgres-data:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-mlflow}"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.17.2
    command: >
      sh -c "pip install --no-cache-dir psycopg2-binary &&
             mlflow server --host 0.0.0.0 --port 5000
             --backend-store-uri postgresql://${POSTGRES_USER:-mlflow}:${POSTGRES_PASSWORD:-mlflow}@postgres:5432/${POSTGRES_DB:-mlflow}
             --artifacts-destination /mlartifacts
             --serve-artifacts"
    ports: ["5000:5000"]
    volumes: ["mlflow-artifacts:/mlartifacts"]
    depends_on:
      postgres: {condition: service_healthy}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://localhost:5000/health')"]
      interval: 15s
      timeout: 5s
      retries: 10
      start_period: 60s
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v3.0.1
    ports: ["9090:9090"]
    volumes:
      - ./monitoring/prometheus:/etc/prometheus:ro
      - prometheus-data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.path=/prometheus
    depends_on:
      api: {condition: service_healthy}
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9090/-/healthy"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports: ["9093:9093"]
    volumes: ["./monitoring/alertmanager:/etc/alertmanager:ro"]
    command: ["--config.file=/etc/alertmanager/alertmanager.yml"]
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9093/-/healthy"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.4.0
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:-admin}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      prometheus: {condition: service_healthy}
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

volumes:
  postgres-data:
  mlflow-artifacts:
  prometheus-data:
  grafana-data:
```

- [ ] **Step 2: Create the Prometheus configuration and alert rules**

`monitoring/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts.yml

alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  - job_name: sentiment-api
    metrics_path: /metrics
    static_configs:
      - targets: ["api:8000"]

  - job_name: prometheus
    static_configs:
      - targets: ["localhost:9090"]
```

`monitoring/prometheus/alerts.yml`:

```yaml
groups:
  - name: sentiment-service
    rules:
      - alert: APIDown
        expr: up{job="sentiment-api"} == 0
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "Sentiment API is not being scraped"
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#apidown"

      - alert: HighErrorRate
        expr: |
          sum(rate(sentiment_requests_total{status=~"5.."}[5m]))
            / clamp_min(sum(rate(sentiment_requests_total[5m])), 0.001) > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "5xx rate above 5% for 5 minutes"
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#higherrorrate"

      - alert: HighLatencyP95
        expr: |
          histogram_quantile(
            0.95, sum(rate(sentiment_request_duration_seconds_bucket[5m])) by (le)
          ) > 0.5
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "p95 latency above 500ms (2.5x the 200ms SLO)"
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#highlatencyp95"

      - alert: PredictionSkew
        expr: |
          abs(
            sum(rate(sentiment_predictions_total{label="positive"}[15m]))
              / clamp_min(sum(rate(sentiment_predictions_total[15m])), 0.001)
            - 0.5
          ) > 0.2
        for: 15m
        labels: {severity: warning}
        annotations:
          summary: "Positive-class rate deviates more than 20pp from the training prior"
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#predictionskew"

      - alert: DriftDetected
        expr: sentiment_drift_psi > 0.2
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Input length PSI above 0.2 (conventional significant-shift boundary)"
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#driftdetected"
```

- [ ] **Step 3: Create the Alertmanager and Grafana provisioning files**

`monitoring/alertmanager/alertmanager.yml`:

```yaml
route:
  receiver: default
  group_by: [alertname, severity]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

receivers:
  - name: default
```

`monitoring/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`monitoring/grafana/provisioning/dashboards/dashboards.yml`:

```yaml
apiVersion: 1
providers:
  - name: sentiment
    orgId: 1
    folder: Sentiment
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
```

Create the dashboard directory so the mount resolves before Week 3 fills it:

```bash
mkdir -p monitoring/grafana/dashboards && touch monitoring/grafana/dashboards/.gitkeep
```

- [ ] **Step 4: Write the smoke test**

`tests/integration/test_stack_smoke.py`:

```python
"""End-to-end checks against a running stack.

Run with: docker compose up -d && pytest tests/integration -m integration
"""

import httpx
import pytest

pytestmark = pytest.mark.integration

API = "http://localhost:8000"
PROM = "http://localhost:9090"


def test_api_is_ready():
    body = httpx.get(f"{API}/ready", timeout=10).json()
    assert body["status"] == "ready"


def test_api_serves_a_prediction():
    body = httpx.post(
        f"{API}/api/v1/predict", json={"text": "Excellent build quality."}, timeout=10
    ).json()
    assert body["label"] in {"positive", "negative"}
    assert body["confidence"] >= 0.5


def test_prometheus_has_scraped_the_api():
    result = httpx.get(
        f"{PROM}/api/v1/query", params={"query": 'up{job="sentiment-api"}'}, timeout=10
    ).json()
    assert result["status"] == "success"
    assert result["data"]["result"], "prometheus has no samples for the api target"
    assert result["data"]["result"][0]["value"][1] == "1"


def test_alert_rules_are_loaded():
    rules = httpx.get(f"{PROM}/api/v1/rules", timeout=10).json()
    names = {
        rule["name"]
        for group in rules["data"]["groups"]
        for rule in group["rules"]
    }
    assert {"APIDown", "HighErrorRate", "DriftDetected"} <= names
```

- [ ] **Step 5: Bring the stack up and run the smoke test**

```bash
cp .env.example .env
docker compose up -d --build
sleep 90
docker compose ps
pytest tests/integration -m integration -v --no-cov
```

Expected: all six services show `healthy`; 4 smoke tests pass.

- [ ] **Step 6: Verify the stack tears down and comes back clean**

```bash
docker compose down -v && docker compose up -d --build && sleep 90 && docker compose ps
```

Expected: all services healthy again from an empty volume set. This is the check
that catches configuration that only works because of leftover local state.

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml monitoring
git commit -m "feat: add six-service compose stack with prometheus alerts"
```

---

### Task 11: Continuous integration

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `README.md` (add the CI badge — created in Task 12; if Task 12 has not
  run yet, add the badge there instead)

**Interfaces:**
- Consumes: `make lint`, `make typecheck`, `make test`, the Dockerfile, the smoke test.
- Produces: a required status check named `ci`.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    name: lint, types, tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install
        run: pip install -e ".[dev]"

      - name: Lint
        run: |
          ruff check src tests
          ruff format --check src tests

      - name: Type check
        run: mypy

      - name: Test with coverage gate
        run: pytest -m "not slow and not integration"

      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage.xml

  container:
    name: build, scan, smoke
    runs-on: ubuntu-latest
    needs: quality
    steps:
      - uses: actions/checkout@v4

      - name: Build API image
        run: docker build -f docker/api.Dockerfile -t sentiment-api:ci .

      - name: Scan image
        uses: aquasecurity/trivy-action@0.28.0
        with:
          image-ref: sentiment-api:ci
          format: table
          exit-code: "1"
          severity: CRITICAL,HIGH
          ignore-unfixed: true

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Start the stack
        run: |
          cp .env.example .env
          docker compose up -d --build

      - name: Wait for the API to become ready
        run: |
          for i in $(seq 1 60); do
            if curl -fsS http://localhost:8000/ready; then exit 0; fi
            sleep 5
          done
          docker compose logs
          exit 1

      - name: Smoke test
        run: |
          pip install httpx pytest
          pytest tests/integration -m integration -v --no-cov

      - name: Dump logs on failure
        if: failure()
        run: docker compose logs
```

- [ ] **Step 2: Verify the workflow parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit and push, then confirm the run is green**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add lint, type, test, build, scan and smoke pipeline"
git push -u origin main
gh run watch
```

Expected: both jobs succeed. Do not proceed to Week 2 with a red pipeline —
a broken CI on day five is a broken CI on day twenty-five.

---

### Task 12: Required documentation

**Files:**
- Create: `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`
- Create: `docs/user-guide.md` (alert runbook stubs referenced by `runbook_url`)

**Interfaces:**
- Consumes: the spec at `docs/superpowers/specs/2026-08-09-sentiment-service-design.md`.
- Produces: the four files the brief lists under "Required files".

- [ ] **Step 1: Write `README.md`**

Sections, in order: title and one-line description; CI and coverage badges;
the architecture diagram from spec §2.1; **Quickstart** (`cp .env.example .env`
then `docker compose up -d`, with the URLs for api/docs/mlflow/prometheus/grafana);
**API examples** as runnable `curl` commands for `/api/v1/predict`,
`/api/v1/predict/batch`, and `/api/v1/model/info`, each with its actual JSON
response; **Development** (`make install`, `make lint`, `make typecheck`,
`make test`); **Troubleshooting** with at least these four entries — port 8000
already in use, `mlflow` unhealthy because postgres was slow to start,
`docker compose down -v` needed after changing provisioning files, and the
dataset download timing out; and links to `ARCHITECTURE.md`, `CONTRIBUTING.md`,
and `docs/user-guide.md`.

- [ ] **Step 2: Write `ARCHITECTURE.md`**

Port spec §2 verbatim — the architecture diagram, the component/responsibility
table, the data flow including the edge-case list, and the full technology
trade-off table. Add a "Current state" note recording that the model is
`StubPredictor` until Week 2, so a reader is never misled about what runs today.

- [ ] **Step 3: Write `CONTRIBUTING.md`**

The role table from spec §7 with **real member names filled in**, plus: branch
naming (`<initials>/<short-description>`), Conventional Commits, the PR checklist
(tests pass, coverage gate holds, lint and mypy clean, docs updated), and the
rule that every member reviews at least one PR per week outside their own area.

- [ ] **Step 4: Write `docs/user-guide.md`**

One `##` section per alert, with an anchor exactly matching the `runbook_url`
fragments from Task 10: `#apidown`, `#higherrorrate`, `#highlatencyp95`,
`#predictionskew`, `#driftdetected`. Each section states what fired, the likely
causes, and the first three diagnostic commands to run. Also cover deploying the
stack, reading each dashboard, and rolling back to a previous model version.

- [ ] **Step 5: Verify every runbook link resolves**

```bash
grep -o 'user-guide.md#[a-z]*' monitoring/prometheus/alerts.yml | sort -u | \
  sed 's/.*#//' | while read -r anchor; do
    grep -qi "^## .*" docs/user-guide.md && \
    grep -qi "$anchor" <(grep '^## ' docs/user-guide.md | tr 'A-Z ' 'a-z-') \
      && echo "ok: $anchor" || echo "MISSING: $anchor"
  done
```

Expected: `ok:` for all five anchors. Fix any `MISSING:` before committing.

- [ ] **Step 6: Replace the `OWNER` placeholder in the alert runbook URLs**

Run: `grep -rn 'github.com/OWNER' monitoring/`
Expected: no results after you substitute your real GitHub org/user. This is the
one intentional placeholder in the plan and it must not survive Task 12.

- [ ] **Step 7: Commit**

```bash
git add README.md ARCHITECTURE.md CONTRIBUTING.md docs/user-guide.md monitoring
git commit -m "docs: add README, architecture, contributing and runbooks"
```

---

## Definition of done for Week 1

- [ ] `docker compose down -v && docker compose up -d --build` yields six healthy services from scratch.
- [ ] `curl -X POST localhost:8000/api/v1/predict -H 'content-type: application/json' -d '{"text":"great"}'` returns a full prediction.
- [ ] Prometheus shows `up{job="sentiment-api"} == 1` and all five alert rules loaded.
- [ ] Grafana is reachable with the Prometheus datasource already provisioned.
- [ ] `pytest -m "not slow"` passes with coverage ≥ 80%.
- [ ] `ruff check`, `ruff format --check`, and `mypy` are all clean.
- [ ] CI is green on `main`.
- [ ] `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/` all exist — the brief's required-files list.
- [ ] Every team member has at least one commit.

## What Week 2 changes

Only two things: `_lifespan` in `app.py` binds a `TransformerPredictor` instead of
`StubPredictor`, and the drift reference is loaded from the MLflow artifact instead of
`_BOOTSTRAP_REFERENCE`. Every endpoint, schema, metric, alert, and test written this week
survives unchanged. That is the point of the walking skeleton.
