"""Reading a donated MLflow tracking database.

The fine-tune ran on Kaggle against MLflow 3.x, which left behind a SQLite file
whose schema this project's pinned 2.17.2 client cannot open — ``MlflowClient``
fails on the migration revision before it reads a single run. The columns worth
having are few and stable across both versions, so they are read directly.

The connection is opened read-only. The donated database is the only surviving
record of that training run, and immutable-by-construction beats
remember-not-to-write-it.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class DonatedRun:
    """One training run recovered from a donated tracking database."""

    run_id: str
    run_name: str
    experiment_name: str
    start_time_ms: int
    params: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def trained_at(self) -> str:
        """Training start as an ISO-8601 UTC string, the form serving reports."""
        return datetime.fromtimestamp(self.start_time_ms / 1000.0, tz=timezone.utc).isoformat()


def _key_values(connection: sqlite3.Connection, table: str, run_id: str) -> dict[str, str]:
    rows = connection.execute(f"SELECT key, value FROM {table} WHERE run_uuid = ?", (run_id,))
    return {str(key): str(value) for key, value in rows}


def read_donated_run(database: Path, run_id: str | None = None) -> DonatedRun:
    """Return one run from ``database``, defaulting to the most recent.

    Args:
        database: Path to the donated ``mlflow.db``.
        run_id: Specific run to read. Omit to take the latest by start time.

    Raises:
        FileNotFoundError: The database does not exist.
        ValueError: The database holds no runs, or none with ``run_id``.
    """
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(f"Donated tracking database not found: {database}")

    connection = sqlite3.connect(f"file:{database}?mode=ro&immutable=1", uri=True)
    try:
        if run_id is None:
            row = connection.execute(
                "SELECT run_uuid, name, experiment_id, start_time "
                "FROM runs ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError(f"{database} contains no runs")
        else:
            row = connection.execute(
                "SELECT run_uuid, name, experiment_id, start_time FROM runs WHERE run_uuid = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"{database} contains no run {run_id!r}")

        found_id, run_name, experiment_id, start_time = row
        experiment = connection.execute(
            "SELECT name FROM experiments WHERE experiment_id = ?", (experiment_id,)
        ).fetchone()
        metrics = {
            str(key): float(value)
            for key, value in connection.execute(
                "SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (found_id,)
            )
        }
        return DonatedRun(
            run_id=str(found_id),
            run_name=str(run_name),
            experiment_name=str(experiment[0]) if experiment else "Default",
            start_time_ms=int(start_time),
            params=_key_values(connection, "params", found_id),
            metrics=metrics,
            tags=_key_values(connection, "tags", found_id),
        )
    finally:
        connection.close()
