from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_history_database(path: Path | str) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS analysis_runs (
            run_id TEXT PRIMARY KEY,
            dataset_name TEXT NOT NULL,
            method_name TEXT NOT NULL,
            availability TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','interrupted')),
            requested_at TEXT NOT NULL,
            started_at TEXT,
            completed_at TEXT,
            error_message TEXT,
            result_json TEXT
        )
    """)
    connection.commit()
    return connection


def create_analysis_run(
    connection: sqlite3.Connection,
    dataset_name: str,
    method_name: str,
    availability: str,
) -> dict[str, object]:
    run_id = uuid4().hex
    requested_at = utc_now()
    connection.execute(
        """
        INSERT INTO analysis_runs (
            run_id, dataset_name, method_name, availability, status, requested_at
        ) VALUES (?, ?, ?, ?, 'queued', ?)
        """,
        (run_id, dataset_name, method_name, availability, requested_at),
    )
    connection.commit()
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "method_name": method_name,
        "availability": availability,
        "status": "queued",
        "requested_at": requested_at,
    }


def mark_analysis_running(connection: sqlite3.Connection, run_id: str) -> None:
    started_at = utc_now()
    connection.execute(
        "UPDATE analysis_runs SET status = 'running', started_at = ? WHERE run_id = ?",
        (started_at, run_id),
    )
    connection.commit()


def save_completed_analysis(
    connection: sqlite3.Connection,
    run_id: str,
    result: Any,
) -> None:
    result_json = json.dumps(result, allow_nan=False)
    completed_at = utc_now()
    connection.execute(
        "UPDATE analysis_runs SET status = 'completed', completed_at = ?, result_json = ? WHERE run_id = ?",
        (completed_at, result_json, run_id),
    )
    connection.commit()


def save_failed_analysis(
    connection: sqlite3.Connection,
    run_id: str,
    message: str,
) -> None:
    completed_at = utc_now()
    connection.execute(
        "UPDATE analysis_runs SET status = 'failed', completed_at = ?, error_message = ? WHERE run_id = ?",
        (completed_at, str(message), run_id),
    )
    connection.commit()


def mark_interrupted_runs(connection: sqlite3.Connection) -> int:
    completed_at = utc_now()
    cursor = connection.execute(
        "UPDATE analysis_runs SET status = 'interrupted', completed_at = ? WHERE status IN ('queued', 'running')",
        (completed_at,),
    )
    count = cursor.rowcount
    connection.commit()
    return count


def get_analysis_run(connection: sqlite3.Connection, run_id: str) -> dict[str, object] | None:
    row = connection.execute(
        """
        SELECT run_id, dataset_name, method_name, availability, status,
               requested_at, started_at, completed_at, error_message, result_json
        FROM analysis_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    record = dict(row)
    result_json = record.pop("result_json", None)
    record["result"] = json.loads(result_json) if result_json is not None else None
    return record


def list_analysis_runs(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT run_id, dataset_name, method_name, availability, status,
               requested_at, started_at, completed_at, error_message
        FROM analysis_runs
        ORDER BY requested_at DESC
        LIMIT 200
        """
    ).fetchall()
    return [dict(row) for row in rows]
