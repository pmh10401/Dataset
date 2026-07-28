from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
import sqlite3
from types import MappingProxyType
from urllib.parse import quote

import numpy as np

from experiments.exp403_408.artifact_io import sha256_file

from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import TrainRun


def database_snapshot(db_path: Path) -> Mapping[str, str]:
    database = Path(db_path)
    candidates = (
        ("db", database),
        ("wal", Path(f"{database}-wal")),
        ("shm", Path(f"{database}-shm")),
    )
    try:
        values = {
            name: sha256_file(path)
            for name, path in candidates
            if path.is_file()
        }
    except OSError as error:
        raise ValueError(f"Exp411 cannot snapshot SQLite artifact: {db_path}") from error
    if "db" not in values:
        raise ValueError(f"Exp411 SQLite database does not exist: {db_path}")
    return MappingProxyType(dict(sorted(values.items())))


def trace_sql_contract(statements: Sequence[str]) -> None:
    for statement in statements:
        normalized = " ".join(str(statement).lower().split())
        if "labels_blob" in normalized or "'test'" in normalized or '"test"' in normalized:
            raise ValueError("Exp411 SQL trace contains forbidden TEST or label access")


def load_train_runs(
    db_path: Path,
    datasets: Sequence[str],
    *,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, tuple[TrainRun, ...]]:
    config.validate()
    requested = tuple(str(dataset) for dataset in datasets)
    if (
        len(requested) != config.expected_datasets
        or len(set(requested)) != len(requested)
        or len({_family_from_dataset(dataset) for dataset in requested})
        != config.expected_families
    ):
        raise ValueError("Exp411 requested SQLite dataset universe mismatch")

    before = database_snapshot(db_path)
    if tuple(before) != config.expected_database_snapshot_keys:
        raise ValueError("Exp411 SQLite snapshot artifact set is not pinned")
    if before["db"] != config.database_sha256:
        raise ValueError("Exp411 SQLite database SHA256 mismatch")

    resolved = Path(db_path).resolve()
    connection = sqlite3.connect(
        f"file:{quote(str(resolved), safe='/')}?mode=ro",
        uri=True,
    )
    trace: list[str] = []
    connection.set_trace_callback(trace.append)
    try:
        if connection.execute("PRAGMA quick_check;").fetchall() != [("ok",)]:
            raise ValueError("Exp411 SQLite PRAGMA quick_check failed")
        result = {
            dataset: _load_one_dataset(connection, dataset)
            for dataset in sorted(requested)
        }
        trace_sql_contract(trace)
    except sqlite3.Error as error:
        raise ValueError("Exp411 SQLite TRAIN query failed") from error
    finally:
        connection.close()

    after = database_snapshot(db_path)
    if dict(before) != dict(after):
        raise ValueError("Exp411 SQLite snapshot changed during TRAIN scan")
    return MappingProxyType(result)


def _load_one_dataset(
    connection: sqlite3.Connection,
    dataset: str,
) -> tuple[TrainRun, ...]:
    identifiers = connection.execute(
        "SELECT id FROM datasets WHERE name = ?",
        (dataset,),
    ).fetchall()
    if len(identifiers) != 1:
        raise ValueError(f"Exp411 SQLite dataset is missing or duplicated: {dataset}")
    dataset_id = int(identifiers[0][0])
    total, normal = connection.execute(
        """
        SELECT COUNT(*), SUM(CASE WHEN label = '0' THEN 1 ELSE 0 END)
        FROM instances
        WHERE dataset_id = ? AND split = 'TRAIN'
        """,
        (dataset_id,),
    ).fetchone()
    if int(total) == 0 or int(normal or 0) != int(total):
        raise ValueError(f"Exp411 SQLite TRAIN labels are invalid: {dataset}")

    rows = connection.execute(
        """
        SELECT instance_index, values_blob
        FROM instances
        WHERE dataset_id = ? AND split = 'TRAIN' AND label = '0'
        ORDER BY instance_index
        """,
        (dataset_id,),
    ).fetchall()
    if len(rows) != int(total):
        raise ValueError(f"Exp411 SQLite TRAIN row count mismatch: {dataset}")
    output: list[TrainRun] = []
    for instance_index, blob in rows:
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise ValueError(f"Exp411 SQLite TRAIN blob is invalid: {dataset}")
        payload = bytes(blob)
        if len(payload) % 4:
            raise ValueError(f"Exp411 SQLite TRAIN blob width is invalid: {dataset}")
        values = np.frombuffer(payload, dtype="<f4")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f"Exp411 SQLite TRAIN values are invalid: {dataset}")
        output.append(
            TrainRun(
                dataset=dataset,
                instance_index=int(instance_index),
                values=tuple(float(value) for value in values),
                values_sha256=sha256(payload).hexdigest(),
            )
        )
    if len({row.instance_index for row in output}) != len(output):
        raise ValueError(f"Exp411 SQLite TRAIN indices are duplicated: {dataset}")
    return tuple(output)


def _family_from_dataset(dataset: str) -> str:
    if dataset.count("_normal_") != 1:
        raise ValueError("Exp411 dataset must contain one _normal_ marker")
    family, suffix = dataset.split("_normal_", 1)
    signed_decimal = suffix.isdecimal() or (
        suffix.startswith("-") and suffix[1:].isdecimal()
    )
    if not family or not signed_decimal:
        raise ValueError("Exp411 dataset family and numeric class suffix are required")
    return family


__all__ = [
    "database_snapshot",
    "load_train_runs",
    "trace_sql_contract",
]
