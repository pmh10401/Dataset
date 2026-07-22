from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.config import DB_PATH
from experiments.exp226_235_hard_alert_quality.contracts import DatabaseAudit, DatasetBundle
from experiments.exp226_235_hard_alert_quality.preprocessing import sanitize_series


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = Path(path).resolve(strict=True)
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def _load_split_values(
    connection: sqlite3.Connection, dataset: str, split: str
) -> tuple[np.ndarray, ...]:
    rows = connection.execute(
        """
        SELECT i.values_blob
        FROM instances AS i
        JOIN datasets AS d ON d.id = i.dataset_id
        WHERE d.name = ? AND i.split = ?
        ORDER BY i.instance_index
        """,
        (str(dataset), str(split)),
    ).fetchall()
    if not rows:
        raise ValueError(f"no {split} rows for dataset={dataset}")
    return tuple(
        sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in rows
    )


def load_values_bundle(
    dataset: str, path: Path = DB_PATH
) -> DatasetBundle:
    connection = _connect_read_only(path)
    try:
        train = _load_split_values(connection, dataset, "TRAIN")
        test = _load_split_values(connection, dataset, "TEST")
    finally:
        connection.close()
    return DatasetBundle(dataset=str(dataset), train_normal=train, test_values=test)


def load_test_labels_for_evaluation(
    dataset: str, path: Path = DB_PATH
) -> np.ndarray:
    connection = _connect_read_only(path)
    try:
        rows = connection.execute(
            """
            SELECT i.label
            FROM instances AS i
            JOIN datasets AS d ON d.id = i.dataset_id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (str(dataset),),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"no TEST evaluation rows for dataset={dataset}")
    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)


def audit_database(path: Path = DB_PATH) -> DatabaseAudit:
    connection = _connect_read_only(path)
    try:
        tables = tuple(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        )
        dataset_count = int(connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0])
        counts = dict(
            connection.execute(
                "SELECT split, COUNT(*) FROM instances GROUP BY split"
            ).fetchall()
        )
    finally:
        connection.close()
    return DatabaseAudit(
        database_path=str(Path(path).resolve()),
        database_sha256=sha256_file(Path(path)),
        dataset_count=dataset_count,
        train_instance_count=int(counts.get("TRAIN", 0)),
        test_instance_count=int(counts.get("TEST", 0)),
        tables=tables,
    )

