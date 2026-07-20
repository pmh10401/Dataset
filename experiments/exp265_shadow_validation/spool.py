from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .contracts import prediction_sha256


class SpoolConflictError(ValueError):
    """Raised when one run ID is reused for a different prediction."""


@dataclass(frozen=True)
class SpoolRow:
    run_id: str
    prediction_sha256: str
    payload: dict[str, object]
    state: str
    attempt_count: int
    last_error: str | None
    receipt: dict[str, object] | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _verified_hash(payload: Mapping[str, object]) -> str:
    stated = payload.get("prediction_sha256")
    if type(stated) is not str:
        raise ValueError("prediction_sha256 is required")
    unsigned = dict(payload)
    unsigned.pop("prediction_sha256", None)
    actual = prediction_sha256(unsigned)
    if actual != stated:
        raise ValueError("prediction_sha256 does not match payload")
    return stated


class ShadowSpool:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma journal_mode = wal")
        connection.execute("pragma foreign_keys = on")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                create table if not exists shadow_predictions (
                    run_id text primary key,
                    prediction_sha256 text not null,
                    payload_json text not null,
                    state text not null check (state in ('pending', 'delivered', 'dead_letter')),
                    attempt_count integer not null default 0 check (attempt_count >= 0),
                    last_error text,
                    receipt_json text,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> SpoolRow:
        return SpoolRow(
            run_id=row["run_id"],
            prediction_sha256=row["prediction_sha256"],
            payload=json.loads(row["payload_json"]),
            state=row["state"],
            attempt_count=row["attempt_count"],
            last_error=row["last_error"],
            receipt=json.loads(row["receipt_json"]) if row["receipt_json"] else None,
        )

    def enqueue(self, payload: Mapping[str, object]) -> SpoolRow:
        run_id = payload.get("run_id")
        stated_hash = payload.get("prediction_sha256")
        if type(run_id) is not str or not run_id:
            raise ValueError("run_id is required")
        if type(stated_hash) is not str:
            raise ValueError("prediction_sha256 is required")
        now = _now()
        with closing(self._connect()) as connection, connection:
            connection.execute("begin immediate")
            existing = connection.execute(
                "select * from shadow_predictions where run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                if existing["prediction_sha256"] != stated_hash:
                    raise SpoolConflictError(f"run_id already has a different prediction: {run_id}")
                _verified_hash(payload)
                return self._row(existing)
            verified_hash = _verified_hash(payload)
            connection.execute(
                """
                insert into shadow_predictions (
                    run_id, prediction_sha256, payload_json, state, created_at, updated_at
                ) values (?, ?, ?, 'pending', ?, ?)
                """,
                (run_id, verified_hash, _canonical(payload), now, now),
            )
            row = connection.execute(
                "select * from shadow_predictions where run_id = ?", (run_id,)
            ).fetchone()
            assert row is not None
            return self._row(row)

    def pending(self, limit: int = 100) -> list[SpoolRow]:
        if type(limit) is not int or limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with closing(self._connect()) as connection, connection:
            rows = connection.execute(
                "select * from shadow_predictions where state = 'pending' order by created_at, run_id limit ?",
                (limit,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def get(self, run_id: str) -> SpoolRow:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "select * from shadow_predictions where run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        return self._row(row)

    def mark_delivered(self, run_id: str, receipt: Mapping[str, object]) -> None:
        with closing(self._connect()) as connection, connection:
            connection.execute("begin immediate")
            cursor = connection.execute(
                """
                update shadow_predictions
                set state = 'delivered', receipt_json = ?, last_error = null,
                    attempt_count = attempt_count + 1, updated_at = ?
                where run_id = ? and state = 'pending'
                """,
                (_canonical(receipt), _now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"pending run not found: {run_id}")

    def mark_failure(self, run_id: str, error: str, *, permanent: bool) -> None:
        state = "dead_letter" if permanent else "pending"
        with closing(self._connect()) as connection, connection:
            connection.execute("begin immediate")
            cursor = connection.execute(
                """
                update shadow_predictions
                set state = ?, last_error = ?, attempt_count = attempt_count + 1, updated_at = ?
                where run_id = ? and state = 'pending'
                """,
                (state, error[:2000], _now(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"pending run not found: {run_id}")
