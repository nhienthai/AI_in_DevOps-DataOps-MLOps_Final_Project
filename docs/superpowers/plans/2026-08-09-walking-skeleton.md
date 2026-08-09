# Walking Skeleton Implementation Plan (Week 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the complete six-service Docker Compose stack, serving predictions from a deterministic stub model, with the data pipeline, Prometheus instrumentation, and green CI all working end-to-end — so that Week 2 only has to swap the stub for a real model.

**Architecture:** A `src/sentiment/` package split into `data/` (ingest, validate, preprocess), `serving/` (FastAPI app, predictor protocol, metrics), and `config.py`. The serving layer depends on a `Predictor` Protocol, not on a concrete model, so Week 2's DistilBERT drops in without touching the API. All ML metrics are computed in-process. Docker Compose runs api, mlflow, postgres, prometheus, grafana, alertmanager with health checks throughout.

**Tech Stack:** Python 3.10/3.11, FastAPI, pydantic v2 + pydantic-settings, prometheus-client, pandas + pyarrow, numpy, HuggingFace `datasets`, pytest + `fastapi.testclient`, flake8 + black + isort, mypy, Docker Compose, GitHub Actions.

## Global Constraints

**Course conventions.** This project follows the structure and naming established
in DDM501 Labs 1-4 and the `stock-signal-mlops` assignment, so lab dashboards, CI
config, and middleware port over instead of being rebuilt. Where this plan and a
lab disagree, the lab wins unless a rationale is given inline.

- Python `>=3.10`; CI runs a 3.10 / 3.11 matrix (Lab 3).
- Package root is `src/sentiment/`; all imports are absolute (`from sentiment.data import ...`).
- Lint and format with **flake8 + black + isort, pinned** — not ruff (Lab 3 toolchain).
- Every public function has type hints and a docstring; mypy must pass on `src/` and `scripts/`.
- Coverage gate `--cov-fail-under=80` is active from Task 1 and never lowered.
- **Metric names use the Lab 4 `http_*` / `ml_*` convention.** Never invent a
  `sentiment_*` prefix — Lab 4's Grafana dashboards query these names.
- `score` is P(positive) in [0,1]; `confidence` is `max(score, 1-score)` in [0.5,1]. Never conflate.
- Labels: `0` = negative, `1` = positive (matches HuggingFace `amazon_polarity`).
- Label strings in API responses: `"positive"` / `"negative"` (lowercase).
- No raw review text in logs — only lengths, hashes, and request ids.
- Settings are read via `sentiment.config.get_settings()`, never `os.environ` directly.
- Env var prefix is `SENTIMENT_`.
- API routes live under `/api/v1`; operational routes (`/health`, `/ready`, `/metrics`) are unprefixed.
- Integration tests use `fastapi.testclient.TestClient` **as a context manager**, so
  the lifespan runs and the model is loaded (Lab 3 `conftest.py`; without it every
  prediction returns 503).
- Paths follow the labs: `Dockerfile` at root, `prometheus/`, `grafana/`,
  `alertmanager/`, `scripts/`, `models/` at root; tests in
  `tests/{unit,integration,data,model}/`, each a package with `__init__.py`.
- Commit after every task. Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`, `ci:`).

---

### Task 1: Project scaffold, tooling, and configuration

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.flake8`, `Makefile`, `.env.example`
- Create: `src/sentiment/__init__.py`, `src/sentiment/config.py`
- Create: `src/sentiment/data/__init__.py`, `src/sentiment/serving/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, and `__init__.py` in each test package
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `sentiment.config.Settings` (pydantic-settings model) and
  `sentiment.config.get_settings() -> Settings` (lru_cached). Every later task
  reads configuration through `get_settings()`.

- [ ] **Step 1: Create `pyproject.toml`**

Configuration only — dependencies live in `requirements*.txt`, as in the labs.

```toml
# =============================================================================
# Project configuration
# DDM501 - Final Project: Sentiment Analysis Service
# Tooling mirrors Lab 3 so the CI pipeline ports over unchanged.
# =============================================================================

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sentiment-service"
version = "0.1.0"
description = "Real-time sentiment analysis service for e-commerce reviews"
readme = "README.md"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 100
skip = [".venv", "__pycache__"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
exclude = ["venv", "__pycache__", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short --strict-markers"
markers = [
    "slow: tests that load or train large models (deselect with -m 'not slow')",
    "integration: tests requiring a running stack",
]
filterwarnings = ["ignore::DeprecationWarning", "ignore::UserWarning"]

[tool.coverage.run]
source = ["src/sentiment"]
omit = ["tests/*", "*/__pycache__/*", "*/__init__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]
show_missing = true
fail_under = 80

[tool.coverage.html]
directory = "htmlcov"
```

- [ ] **Step 2: Create the dependency files and flake8 config**

`requirements.txt` (runtime only — this is what ships in the image):

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
pydantic==2.10.4
pydantic-settings==2.7.0
prometheus-client==0.21.1
pandas==2.2.3
pyarrow==18.1.0
numpy==1.26.4
datasets==3.2.0
```

`requirements-dev.txt`:

```
-r requirements.txt
pytest==8.3.4
pytest-cov==6.0.0
httpx==0.28.1
flake8==7.0.0
black==23.12.1
isort==5.13.2
mypy==1.8.0
pandas-stubs==2.3.3.260113
types-requests
```

The lint versions are pinned to exactly what Lab 3's CI installs. Pinning matters
here for a specific reason: an unpinned `black` or `pandas-stubs` release can turn
the build red with no code change, which is the most demoralising possible CI
failure for a five-person team.

`.flake8`:

```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv,venv,build,dist,notebooks
per-file-ignores =
    tests/*:D
```

- [ ] **Step 3: Create the package and test skeletons, and the Makefile**

```bash
mkdir -p src/sentiment/{data,models,training,serving,responsible} \
         tests/{unit,integration,data,model} scripts
touch src/sentiment/__init__.py \
      src/sentiment/data/__init__.py src/sentiment/models/__init__.py \
      src/sentiment/training/__init__.py src/sentiment/serving/__init__.py \
      src/sentiment/responsible/__init__.py \
      tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py \
      tests/data/__init__.py tests/model/__init__.py
```

`tests/conftest.py` — the shared client fixture every integration test uses.
It follows Lab 3: the `TestClient` is a context manager so the lifespan runs.

```python
"""Shared pytest fixtures for all tests."""

from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from sentiment.serving.app import create_app


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Test client with the application lifespan actually running.

    Used as a context manager deliberately: without it the lifespan never
    fires, the predictor stays None, and every prediction returns 503.
    """
    with TestClient(create_app()) as test_client:
        yield test_client
```

`Makefile`:

```makefile
.PHONY: install lint format typecheck test up down smoke

install:
	pip install -e . && pip install -r requirements-dev.txt

lint:
	flake8 src tests scripts
	black --check --diff src tests scripts
	isort --check-only --diff src tests scripts

format:
	black src tests scripts
	isort src tests scripts

typecheck:
	mypy src scripts

test:
	pytest -m "not slow and not integration" \
	  --cov=sentiment --cov-report=term-missing --cov-report=xml --cov-fail-under=80

up:
	docker compose up -d

down:
	docker compose down -v

smoke:
	pytest tests/integration -m integration -v --no-cov
```

- [ ] **Step 4: Write the failing test for configuration**

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

- [ ] **Step 5: Run the test to verify it fails**

Run: `pytest tests/unit/test_config.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'sentiment.config'`

- [ ] **Step 6: Implement `src/sentiment/config.py`**

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

- [ ] **Step 7: Create `.env.example`**

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

- [ ] **Step 8: Run tests and the full lint suite to verify they pass**

```bash
make install
pytest tests/unit/test_config.py -v --no-cov
make lint
make typecheck
```

Expected: 3 passed, flake8/black/isort clean, mypy clean.
If black reports reformatting, run `make format` and re-check — do not hand-fix
formatting, the whole point of black is that nobody argues about it.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt .flake8 Makefile .env.example src tests
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
- Test: `tests/data/test_validate.py`

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

`tests/data/test_validate.py`:

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

Run: `pytest tests/data/test_validate.py -v --no-cov`
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

Run: `pytest tests/data/test_validate.py -v --no-cov`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sentiment/data/validate.py tests/data/test_validate.py
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
  - Module-level collectors, keeping Lab 4's names so its Grafana dashboards work:
    `REQUEST_COUNT`, `REQUEST_LATENCY`, `PREDICTION_COUNT`, `PREDICTION_LATENCY`,
    `PREDICTION_ERRORS`, `MODEL_LOADED`, `MODEL_INFO`, `MODEL_LAST_RELOAD`,
    `BATCH_SIZE`, plus this project's additions `PREDICTION_CONFIDENCE`,
    `LOW_CONFIDENCE`, `INPUT_LENGTH`, `DRIFT_PSI`, `FAIRNESS_MAX_DELTA`.

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
from prometheus_client import Counter, Gauge, Histogram, Info

from sentiment.data.preprocess import DriftReference

MIN_OBSERVATIONS = 30

# ---------------------------------------------------------------------------
# HTTP metrics. Names and labels are Lab 4's, unchanged, so that lab's
# system_dashboard.json imports and works without editing a single query.
# Recorded by MetricsMiddleware, never per-endpoint.
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ---------------------------------------------------------------------------
# ML metrics. Lab 4 names, with one deliberate change: PREDICTION_COUNT gains a
# `label` dimension, because class skew is a real signal for a classifier
# whereas Lab 4 predicted a continuous rating.
# ---------------------------------------------------------------------------
PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["label", "model_version"],
)
PREDICTION_LATENCY = Histogram(
    "ml_prediction_duration_seconds",
    "Time to generate a prediction",
    ["model_version"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)
PREDICTION_ERRORS = Counter(
    "ml_prediction_errors_total",
    "Total number of prediction errors",
    ["error_type", "model_version"],
)
MODEL_LOADED = Gauge(
    "ml_model_loaded",
    "Whether the ML model is loaded (1) or not (0)",
)
MODEL_INFO = Info(
    "ml_model_info",
    "Information about the loaded ML model",
)
MODEL_LAST_RELOAD = Gauge(
    "ml_model_last_reload_timestamp",
    "Unix timestamp of last model reload",
)
BATCH_SIZE = Histogram(
    "ml_batch_prediction_size",
    "Size of batch prediction requests",
    buckets=[1, 5, 10, 25, 50, 100],
)

# ---------------------------------------------------------------------------
# Custom metrics added by this project. Each exists because a specific alert or
# dashboard panel needs it; none is here to pad the count.
# ---------------------------------------------------------------------------
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribution of prediction confidence",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
LOW_CONFIDENCE = Counter(
    "ml_low_confidence_total",
    "Predictions below the low-confidence threshold",
)
INPUT_LENGTH = Histogram(
    "ml_input_length_chars",
    "Distribution of input text length in characters",
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
)
DRIFT_PSI = Gauge(
    "ml_drift_psi",
    "Population stability index of live input lengths against the training reference",
)
FAIRNESS_MAX_DELTA = Gauge(
    "ml_fairness_max_delta",
    "Maximum EEC identity-pair score delta measured for the served model",
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
- Create: `src/sentiment/serving/schemas.py`, `src/sentiment/serving/errors.py`, `src/sentiment/serving/middleware.py`, `src/sentiment/serving/app.py`
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

The `client` fixture comes from `tests/conftest.py` (Task 1) — it is a
`TestClient` used as a context manager, so the lifespan runs and the predictor
is loaded.

```python
from fastapi.testclient import TestClient


def test_health_is_up_and_independent_of_the_model(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_reports_the_loaded_model(client: TestClient):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_version"] == "stub-0"


def test_metrics_endpoint_exposes_prometheus_text(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_model_loaded_gauge_is_set(client: TestClient):
    assert "ml_model_loaded 1.0" in client.get("/metrics").text


def test_openapi_spec_is_generated(client: TestClient):
    spec = client.get("/openapi.json").json()
    assert spec["info"]["title"] == "Sentiment Service"
    assert "/ready" in spec["paths"]


def test_unknown_route_returns_typed_error(client: TestClient):
    response = client.get("/api/v1/nope")
    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "not_found"
    assert "request_id" in body


def test_every_response_carries_a_request_id_header(client: TestClient):
    assert client.get("/health").headers["x-request-id"]
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

- [ ] **Step 5: Implement `src/sentiment/serving/middleware.py`**

This is Lab 4's `MetricsMiddleware`, adapted to also carry the request id. HTTP
metrics live here rather than in each endpoint so that coverage cannot drift as
routes are added.

```python
"""Middleware collecting HTTP metrics and tagging each request with an id."""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from sentiment.serving.metrics import REQUEST_COUNT, REQUEST_LATENCY


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Time the request, label it by route template, and record metrics."""
        request.state.request_id = str(uuid.uuid4())

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # The route template, not request.url.path: labelling by raw path would
        # create a new time series per distinct URL and blow up cardinality.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)

        REQUEST_COUNT.labels(
            method=request.method, endpoint=endpoint, status=str(response.status_code)
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)

        response.headers["x-request-id"] = request.state.request_id
        return response
```

- [ ] **Step 6: Implement `src/sentiment/serving/app.py` with health, ready, and metrics**

```python
"""The FastAPI application factory."""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from sentiment.config import get_settings
from sentiment.data.preprocess import DriftReference
from sentiment.serving.errors import install_error_handlers
from sentiment.serving.metrics import (
    MODEL_INFO,
    MODEL_LAST_RELOAD,
    MODEL_LOADED,
    DriftTracker,
)
from sentiment.serving.middleware import MetricsMiddleware
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
    MODEL_LOADED.set(1)
    MODEL_INFO.info(
        {
            "version": app.state.predictor.version,
            "predictor_class": type(app.state.predictor).__name__,
            "stage": settings.model_stage,
        }
    )
    MODEL_LAST_RELOAD.set(time.time())
    yield
    app.state.predictor = None
    MODEL_LOADED.set(0)


def create_app() -> FastAPI:
    """Build the application. Used by tests and by uvicorn --factory."""
    app = FastAPI(
        title="Sentiment Service",
        version="0.1.0",
        description="Real-time sentiment analysis for e-commerce reviews.",
        lifespan=_lifespan,
    )
    app.add_middleware(MetricsMiddleware)
    install_error_handlers(app)

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

- [ ] **Step 7: Run the tests to verify they pass**

Run: `pytest tests/integration/test_app_health.py -v --no-cov`
Expected: 7 passed.

- [ ] **Step 8: Commit**

```bash
git add src/sentiment/serving/ tests/integration/test_app_health.py
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
import pytest
from fastapi.testclient import TestClient


def test_predict_returns_the_full_contract(client: TestClient):
    body = client.post("/api/v1/predict", json={"text": "Great product"}).json()
    assert set(body) == {
        "label", "score", "confidence", "model_version", "truncated", "latency_ms"
    }


def test_confidence_is_derived_from_score(client: TestClient):
    body = client.post("/api/v1/predict", json={"text": "Great product"}).json()
    assert body["confidence"] == pytest.approx(max(body["score"], 1 - body["score"]))


def test_blank_text_is_rejected(client: TestClient):
    response = client.post("/api/v1/predict", json={"text": "   "})
    assert response.status_code == 422
    assert response.json()["error_code"] in {"validation_error", "empty_text"}


def test_missing_field_is_rejected(client: TestClient):
    assert client.post("/api/v1/predict", json={}).status_code == 422


def test_oversized_text_returns_413(client: TestClient):
    response = client.post("/api/v1/predict", json={"text": "x" * 100_000})
    assert response.status_code == 413
    assert response.json()["error_code"] == "text_too_long"


def test_batch_returns_one_result_per_input(client: TestClient):
    body = client.post("/api/v1/predict/batch", json={"texts": ["a", "b", "c"]}).json()
    assert [r["index"] for r in body["results"]] == [0, 1, 2]
    assert all(r["error"] is None for r in body["results"])


def test_batch_isolates_per_item_failures(client: TestClient):
    body = client.post("/api/v1/predict/batch", json={"texts": ["ok", "  "]}).json()
    assert body["results"][0]["prediction"] is not None
    assert body["results"][1]["error"] is not None


def test_batch_over_the_limit_is_rejected(client: TestClient):
    response = client.post("/api/v1/predict/batch", json={"texts": ["x"] * 65})
    assert response.status_code == 413
    assert response.json()["error_code"] == "batch_too_large"


def test_model_info_reports_provenance(client: TestClient):
    body = client.get("/api/v1/model/info").json()
    assert body["model_version"] == "stub-0"
    assert body["predictor_class"] == "StubPredictor"


def test_predictions_are_counted_by_label(client: TestClient):
    client.post("/api/v1/predict", json={"text": "counted"})
    text = client.get("/metrics").text
    assert "ml_predictions_total" in text
    assert 'model_version="stub-0"' in text


def test_input_length_is_observed(client: TestClient):
    client.post("/api/v1/predict", json={"text": "measured"})
    assert "ml_input_length_chars_count" in client.get("/metrics").text


def test_confidence_is_observed(client: TestClient):
    client.post("/api/v1/predict", json={"text": "measured"})
    assert "ml_prediction_confidence_count" in client.get("/metrics").text


def test_batch_size_is_observed(client: TestClient):
    client.post("/api/v1/predict/batch", json={"texts": ["a", "b"]})
    assert "ml_batch_prediction_size_count" in client.get("/metrics").text


def test_per_item_failure_is_counted_as_a_prediction_error(client: TestClient):
    client.post("/api/v1/predict/batch", json={"texts": ["  "]})
    assert "ml_prediction_errors_total" in client.get("/metrics").text


def test_http_metrics_label_by_route_template_not_raw_path(client: TestClient):
    client.post("/api/v1/predict", json={"text": "labelled"})
    text = client.get("/metrics").text
    assert 'endpoint="/api/v1/predict"' in text


def test_inference_routes_appear_in_the_openapi_spec(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/predict", "/api/v1/predict/batch", "/api/v1/model/info"} <= set(paths)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/integration/test_predict.py -v --no-cov`
Expected: FAIL — every prediction call returns 404.

- [ ] **Step 3: Add the router to `create_app` in `src/sentiment/serving/app.py`**

Note there is no HTTP instrumentation here — `MetricsMiddleware` already records
`http_requests_total` and `http_request_duration_seconds` for every route. Only
ML metrics belong in this layer, where the prediction is in scope.

Insert these imports at the top of the module (`time` is already imported):

```python
from fastapi import APIRouter

from sentiment.serving.errors import APIError
from sentiment.serving.metrics import (
    BATCH_SIZE,
    DRIFT_PSI,
    INPUT_LENGTH,
    LOW_CONFIDENCE,
    PREDICTION_CONFIDENCE,
    PREDICTION_COUNT,
    PREDICTION_ERRORS,
    PREDICTION_LATENCY,
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
        version = request.app.state.predictor.version

        if not text.strip():
            PREDICTION_ERRORS.labels(
                error_type="empty_text", model_version=version
            ).inc()
            raise APIError(422, "empty_text", "Text must not be blank.")
        if len(text) > settings.max_text_length:
            PREDICTION_ERRORS.labels(
                error_type="text_too_long", model_version=version
            ).inc()
            raise APIError(
                413,
                "text_too_long",
                f"Text exceeds {settings.max_text_length} characters.",
            )

        started = time.perf_counter()
        prediction = request.app.state.predictor.predict([text])[0]
        elapsed = time.perf_counter() - started

        PREDICTION_LATENCY.labels(model_version=version).observe(elapsed)
        PREDICTION_COUNT.labels(label=prediction.label, model_version=version).inc()
        PREDICTION_CONFIDENCE.observe(prediction.confidence)
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
        prediction, latency_ms = _score_one(request, payload.text)
        return _to_response(request, prediction, latency_ms)

    @router.post("/predict/batch", response_model=BatchResponse)
    async def predict_batch(request: Request, payload: BatchRequest) -> BatchResponse:
        """Classify several texts, isolating per-item failures."""
        version = request.app.state.predictor.version
        if len(payload.texts) > settings.max_batch_size:
            PREDICTION_ERRORS.labels(
                error_type="batch_too_large", model_version=version
            ).inc()
            raise APIError(
                413,
                "batch_too_large",
                f"Batch exceeds {settings.max_batch_size} items.",
            )

        BATCH_SIZE.observe(len(payload.texts))

        results: list[BatchItem] = []
        for index, text in enumerate(payload.texts):
            try:
                prediction, latency_ms = _score_one(request, text)
            except APIError as exc:
                # A bad item is not a bad request: record it and carry on, so one
                # malformed review cannot fail the other sixty-three.
                results.append(BatchItem(index=index, error=exc.message))
            else:
                results.append(
                    BatchItem(
                        index=index,
                        prediction=_to_response(request, prediction, latency_ms),
                    )
                )
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

Run: `pytest tests/ -v --no-cov && make lint && make typecheck`
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
- Create: `Dockerfile`, `.dockerignore`
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
.mypy_cache
htmlcov
```

- [ ] **Step 2: Create `Dockerfile`**

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

Run: `docker build -t sentiment-api:dev .`
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
git add Dockerfile .dockerignore
git commit -m "feat: add multi-stage non-root API image"
```

---

### Task 10: Compose stack, monitoring configuration, and smoke test

**Files:**
- Create: `docker-compose.yml`
- Create: `prometheus/prometheus.yml`, `prometheus/alerts/api_alerts.yml`, `prometheus/alerts/ml_alerts.yml`
- Create: `alertmanager/alertmanager.yml`
- Create: `grafana/provisioning/datasources/prometheus.yml`
- Create: `grafana/provisioning/dashboards/dashboards.yml`
- Test: `tests/integration/test_stack_smoke.py`

**Interfaces:**
- Consumes: the API image from Task 9, the metric names from Task 6.
- Produces: a six-service stack. Week 3 adds dashboard JSON under
  `grafana/dashboards/`, which the provisioning file already points at.

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
name: sentiment-service

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
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
      - ./prometheus:/etc/prometheus:ro
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
    volumes: ["./alertmanager:/etc/alertmanager:ro"]
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
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
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

`prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - /etc/prometheus/alerts/*.yml

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

Alerts are split by concern into two files, as in Lab 4. Week 3 adds a third,
`fairness_alerts.yml`, which the `alerts/*.yml` glob already picks up.

`prometheus/alerts/api_alerts.yml`:

```yaml
groups:
  - name: api_alerts
    rules:
      - alert: APIDown
        expr: up{job="sentiment-api"} == 0
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "Sentiment API is not being scraped"
          description: "Prometheus cannot reach the API target."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#apidown"

      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m]))
            / clamp_min(sum(rate(http_requests_total[5m])), 0.001) > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "5xx rate above 5% for 5 minutes"
          description: "{{ $value | humanizePercentage }} of requests are failing."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#higherrorrate"

      - alert: HighLatencyP95
        expr: |
          histogram_quantile(
            0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 0.5
        for: 5m
        labels: {severity: warning}
        annotations:
          summary: "p95 latency above 500ms (2.5x the 200ms SLO)"
          description: "P95 latency is {{ $value }}s."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#highlatencyp95"
```

`prometheus/alerts/ml_alerts.yml` — the first, fourth, and fifth rules are Lab 4's,
reused unchanged because the metric names match:

```yaml
groups:
  - name: ml_alerts
    rules:
      - alert: ModelNotLoaded
        expr: ml_model_loaded == 0
        for: 1m
        labels: {severity: critical}
        annotations:
          summary: "ML model not loaded"
          description: "The API is up but no model is loaded; predictions return 503."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#modelnotloaded"

      - alert: PredictionSkew
        expr: |
          abs(
            sum(rate(ml_predictions_total{label="positive"}[15m]))
              / clamp_min(sum(rate(ml_predictions_total[15m])), 0.001)
            - 0.5
          ) > 0.2
        for: 15m
        labels: {severity: warning}
        annotations:
          summary: "Positive-class rate deviates more than 20pp from the training prior"
          description: "Positive share has drifted to {{ $value | humanizePercentage }} away from 0.5."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#predictionskew"

      - alert: DriftDetected
        expr: ml_drift_psi > 0.2
        for: 10m
        labels: {severity: warning}
        annotations:
          summary: "Input length PSI above 0.2 (conventional significant-shift boundary)"
          description: "PSI is {{ $value }} against the training reference."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#driftdetected"

      - alert: HighPredictionErrorRate
        expr: |
          rate(ml_prediction_errors_total[5m])
            / clamp_min(rate(ml_predictions_total[5m]), 0.000001) > 0.05
        for: 5m
        labels: {severity: critical}
        annotations:
          summary: "High prediction error rate"
          description: "{{ $value | humanizePercentage }} of predictions are failing."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#highpredictionerrorrate"

      - alert: ModelStale
        expr: time() - ml_model_last_reload_timestamp > 604800
        for: 1h
        labels: {severity: info}
        annotations:
          summary: "ML model may be stale"
          description: "Model has not been reloaded in {{ $value | humanizeDuration }}."
          runbook_url: "https://github.com/OWNER/sentiment-service/blob/main/docs/user-guide.md#modelstale"
```

- [ ] **Step 3: Create the Alertmanager and Grafana provisioning files**

`alertmanager/alertmanager.yml`:

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

`grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

`grafana/provisioning/dashboards/dashboards.yml`:

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
mkdir -p grafana/dashboards && touch grafana/dashboards/.gitkeep
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

The pipeline shape is Lab 3's, kept deliberately: cheap checks gate expensive
ones, and a single aggregate `ci-status` job is what branch protection requires.

```yaml
# =============================================================================
# CI Pipeline for Sentiment Service
# DDM501 - Final Project
#
#   lint ─┐
#         ├─> test (3.10, 3.11) ─> container (build, scan, smoke) ─> ci-status
#   type ─┘
#
# Lint and type checks finish in seconds and stop a bad commit before anyone
# pays for a Docker build.
# =============================================================================

name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

env:
  PYTHON_DEFAULT: "3.10"

jobs:
  lint:
    name: Lint Code
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT }}
          cache: pip
          cache-dependency-path: requirements-dev.txt

      - name: Install linters
        # Pinned to the same versions as requirements-dev.txt: an unpinned
        # release can turn the build red with no code change.
        run: |
          python -m pip install --upgrade pip
          pip install flake8==7.0.0 black==23.12.1 isort==5.13.2

      - name: Run flake8
        run: flake8 src/ tests/ scripts/

      - name: Check black formatting
        run: black --check --diff src/ tests/ scripts/

      - name: Check import sorting
        run: isort --check-only --diff src/ tests/ scripts/

  type-check:
    name: Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT }}
          cache: pip

      - name: Install dependencies
        # Derived from requirements.txt rather than hand-listed, so a new
        # runtime dependency can never be missing here.
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install mypy==1.8.0 types-requests pandas-stubs==2.3.3.260113
          pip install -e .

      - name: Run mypy
        run: mypy src/ scripts/ --ignore-missing-imports

  test:
    name: Run Tests (py${{ matrix.python-version }})
    runs-on: ubuntu-latest
    needs: [lint, type-check]
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: pip
          cache-dependency-path: |
            requirements.txt
            requirements-dev.txt

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt
          pip install -e .

      - name: Run tests with coverage
        run: |
          pytest tests/ -v \
            -m "not slow and not integration" \
            --cov=sentiment \
            --cov-report=term-missing \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=80 \
            --junitxml=junit.xml

      - name: Upload coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report-py${{ matrix.python-version }}
          path: |
            htmlcov/
            coverage.xml
          retention-days: 14

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results-py${{ matrix.python-version }}
          path: junit.xml
          retention-days: 14

  container:
    name: Build, Scan, Smoke
    runs-on: ubuntu-latest
    needs: [test]
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build API image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: false
          load: true
          tags: sentiment-api:ci
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Scan image
        uses: aquasecurity/trivy-action@0.28.0
        with:
          image-ref: sentiment-api:ci
          format: table
          exit-code: "1"
          severity: CRITICAL,HIGH
          ignore-unfixed: true

      - name: Start the stack
        run: |
          cp .env.example .env
          docker compose up -d --build

      - name: Wait for the API to become ready
        # /ready, not /health: a container that is up but serving 503 because no
        # model loaded is still a broken release.
        run: |
          for _ in $(seq 1 60); do
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

  ci-status:
    name: CI Status
    runs-on: ubuntu-latest
    needs: [lint, type-check, test, container]
    if: always()
    steps:
      - name: Fail if any job failed
        if: contains(needs.*.result, 'failure') || contains(needs.*.result, 'cancelled')
        run: |
          echo "One or more CI jobs failed."
          exit 1

      - name: Report success
        run: echo "All CI checks passed."
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
- Create: `docs/TESTING_STRATEGY.md` (the four test types and what each is for, as in Lab 3)

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
`#modelnotloaded`, `#predictionskew`, `#driftdetected`,
`#highpredictionerrorrate`, `#modelstale`. Each section states what fired, the
likely causes, and the first three diagnostic commands to run. Also cover
deploying the stack, reading each dashboard, and rolling back to a previous
model version.

- [ ] **Step 5: Verify every runbook link resolves**

A runbook link that 404s at 3am is worse than no link, so this is checked
mechanically rather than by eye.

```bash
grep -ho 'user-guide.md#[a-z]*' prometheus/alerts/*.yml | sed 's/.*#//' | sort -u | \
  while read -r anchor; do
    if grep '^## ' docs/user-guide.md | tr 'A-Z ' 'a-z-' | grep -q "$anchor"; then
      echo "ok: $anchor"
    else
      echo "MISSING: $anchor"
    fi
  done
```

Expected: `ok:` for all eight anchors. Fix any `MISSING:` before committing.

- [ ] **Step 6: Replace the `OWNER` placeholder in the alert runbook URLs**

Run: `grep -rn 'github.com/OWNER' prometheus/`
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
- [ ] Prometheus shows `up{job="sentiment-api"} == 1` and all eight alert rules loaded.
- [ ] `/metrics` exposes both `http_*` and `ml_*` families — Lab 4's dashboards depend on these exact names.
- [ ] Grafana is reachable with the Prometheus datasource already provisioned.
- [ ] `pytest -m "not slow"` passes with coverage ≥ 80%.
- [ ] `make lint` (flake8 + black + isort) and `make typecheck` (mypy) are clean.
- [ ] CI is green on `main`.
- [ ] `README.md`, `ARCHITECTURE.md`, `CONTRIBUTING.md`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `.github/workflows/` all exist — the brief's required-files list.
- [ ] Every team member has at least one commit.

## What Week 2 changes

Only two things: `_lifespan` in `app.py` binds a `TransformerPredictor` instead of
`StubPredictor`, and the drift reference is loaded from the MLflow artifact instead of
`_BOOTSTRAP_REFERENCE`. Every endpoint, schema, metric, alert, and test written this week
survives unchanged. That is the point of the walking skeleton.
