"""Reading the donated MLflow 3.x tracking database with the 2.17.2 client pinned."""

import sqlite3
import stat
from pathlib import Path

import pytest

from sentiment.models.donated import read_donated_run


def _build_database(path: Path, *, runs: int = 1) -> None:
    """Create the subset of MLflow's schema the reader touches."""
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE experiments (
            experiment_id INTEGER, name TEXT, artifact_location TEXT, lifecycle_stage TEXT
        );
        CREATE TABLE runs (
            run_uuid TEXT, name TEXT, experiment_id INTEGER,
            start_time INTEGER, artifact_uri TEXT, status TEXT
        );
        CREATE TABLE latest_metrics (run_uuid TEXT, key TEXT, value REAL, step INTEGER);
        CREATE TABLE params (run_uuid TEXT, key TEXT, value TEXT);
        CREATE TABLE tags (run_uuid TEXT, key TEXT, value TEXT);
        """
    )
    connection.execute(
        "INSERT INTO experiments VALUES (1, 'sentiment-analysis-uit-vsfc', '/kaggle/mlruns/1', "
        "'active')"
    )
    for index in range(runs):
        run_id = f"run{index}"
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, 1, ?, '/kaggle/artifacts', 'FINISHED')",
            (run_id, f"fine-tune-{index}", 1_786_551_079_295 + index),
        )
        connection.execute(
            "INSERT INTO latest_metrics VALUES (?, 'test_macro_f1', ?, 0)", (run_id, 0.8 + index)
        )
        connection.execute("INSERT INTO params VALUES (?, 'epochs', '10')", (run_id,))
        connection.execute(
            "INSERT INTO tags VALUES (?, 'mlflow.source.git.commit', 'deadbeef')", (run_id,)
        )
    connection.commit()
    connection.close()


def test_reads_run_identity_metrics_and_params(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database)

    run = read_donated_run(database)

    assert run.run_id == "run0"
    assert run.run_name == "fine-tune-0"
    assert run.experiment_name == "sentiment-analysis-uit-vsfc"
    assert run.metrics == {"test_macro_f1": pytest.approx(0.8)}
    assert run.params == {"epochs": "10"}
    assert run.tags["mlflow.source.git.commit"] == "deadbeef"
    assert run.start_time_ms == 1_786_551_079_295


def test_latest_run_wins_when_the_database_holds_several(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database, runs=3)

    assert read_donated_run(database).run_id == "run2"


def test_named_run_can_be_selected(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database, runs=3)

    assert read_donated_run(database, run_id="run1").run_name == "fine-tune-1"


def test_unknown_run_id_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database)

    with pytest.raises(ValueError, match="absent"):
        read_donated_run(database, run_id="absent")


def test_empty_database_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database, runs=0)

    with pytest.raises(ValueError, match="no runs"):
        read_donated_run(database)


def test_missing_database_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_donated_run(tmp_path / "absent.db")


def test_donated_database_is_opened_read_only(tmp_path: Path) -> None:
    """The donated file is evidence; reading it must never rewrite it."""
    database = tmp_path / "mlflow.db"
    _build_database(database)
    database.chmod(stat.S_IRUSR)

    run = read_donated_run(database)

    assert run.run_id == "run0"
    assert not (tmp_path / "mlflow.db-wal").exists()


def test_trained_at_is_iso_utc(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    _build_database(database)

    assert read_donated_run(database).trained_at.startswith("2026-08-")
