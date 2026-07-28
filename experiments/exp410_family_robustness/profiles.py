"""Read-only SQLite profile scanning for the Exp410 audit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
import sqlite3
from types import MappingProxyType
from urllib.parse import quote

from experiments.exp403_408.artifact_io import sha256_file

from .config import FROZEN_CONFIG, Exp410FrozenConfig
from .contracts import DatasetProfile, family_from_dataset


def database_snapshot(db_path: Path) -> Mapping[str, str]:
    """Hash the DB and any existing WAL/SHM siblings in stable key order."""
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
        raise ValueError(f"Exp410 cannot snapshot SQLite artifact: {db_path}") from error
    if "db" not in values:
        raise ValueError(f"Exp410 SQLite database does not exist: {db_path}")
    return MappingProxyType(dict(sorted(values.items())))


def scan_dataset_profiles(
    db_path: Path,
    datasets: Sequence[str],
    config: Exp410FrozenConfig = FROZEN_CONFIG,
    *,
    expected_snapshot: Mapping[str, str] | None = None,
) -> tuple[DatasetProfile, ...]:
    """Stream exact TRAIN/TEST profiles without materializing the database."""
    requested = tuple(str(dataset) for dataset in datasets)
    if (
        len(requested) != config.expected_datasets
        or len(set(requested)) != len(requested)
        or len({family_from_dataset(dataset) for dataset in requested})
        != config.expected_families
    ):
        raise ValueError("Exp410 requested SQLite dataset universe mismatch")
    ordered = tuple(sorted(requested))

    before = database_snapshot(db_path)
    if tuple(before) != config.expected_database_snapshot_keys:
        raise ValueError("Exp410 SQLite snapshot artifact set is not pinned")
    if expected_snapshot is not None and dict(before) != dict(expected_snapshot):
        raise ValueError("Exp410 SQLite preflight snapshot mismatch")
    if before.get("db") != config.database_sha256:
        raise ValueError("Exp410 SQLite database SHA256 mismatch")

    resolved = Path(db_path).resolve()
    uri = f"file:{quote(str(resolved), safe='/')}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as error:
        raise ValueError(f"Exp410 cannot open SQLite database read-only: {db_path}") from error

    try:
        quick_check = connection.execute("PRAGMA quick_check;").fetchall()
        if quick_check != [("ok",)]:
            raise ValueError("Exp410 SQLite PRAGMA quick_check failed")
        rows = tuple(_scan_one(connection, dataset) for dataset in ordered)
    except sqlite3.Error as error:
        raise ValueError("Exp410 SQLite profile query failed") from error
    finally:
        connection.close()

    after = database_snapshot(db_path)
    if dict(before) != dict(after):
        raise ValueError("Exp410 SQLite snapshot changed during profile scan")
    if tuple(row.dataset for row in rows) != ordered:
        raise ValueError("Exp410 scanned SQLite dataset universe mismatch")
    return rows


def _scan_one(connection: sqlite3.Connection, dataset: str) -> DatasetProfile:
    identifiers = connection.execute(
        "SELECT id FROM datasets WHERE name = ?",
        (dataset,),
    ).fetchall()
    if len(identifiers) != 1:
        raise ValueError(f"Exp410 SQLite dataset is missing or duplicated: {dataset}")
    dataset_id = int(identifiers[0][0])

    train_hashes: set[bytes] = set()
    train_count = 0
    train_rows = connection.execute(
        """
        SELECT label, values_blob
        FROM instances
        WHERE dataset_id = ? AND split = 'TRAIN'
        ORDER BY instance_index
        """,
        (dataset_id,),
    )
    for label, blob in train_rows:
        if label != "0":
            raise ValueError(f"Exp410 SQLite TRAIN label is invalid: {dataset}")
        train_count += 1
        train_hashes.add(_blob_digest(blob, dataset))
    if train_count == 0:
        raise ValueError(f"Exp410 SQLite dataset has zero normal TRAIN rows: {dataset}")

    test_total = 0
    test_anomaly = 0
    test_normal = 0
    exact_overlap = 0
    test_rows = connection.execute(
        """
        SELECT label, values_blob
        FROM instances
        WHERE dataset_id = ? AND split = 'TEST'
        ORDER BY instance_index
        """,
        (dataset_id,),
    )
    for label, blob in test_rows:
        test_total += 1
        if label == "1":
            test_anomaly += 1
        elif label == "0":
            test_normal += 1
            exact_overlap += int(_blob_digest(blob, dataset) in train_hashes)
        else:
            raise ValueError(f"Exp410 SQLite TEST label is invalid: {dataset}")
    if test_anomaly == 0:
        raise ValueError(f"Exp410 SQLite dataset has zero TEST anomalies: {dataset}")

    return DatasetProfile(
        dataset=dataset,
        family=family_from_dataset(dataset),
        train_normal_count=train_count,
        test_total_count=test_total,
        test_anomaly_count=test_anomaly,
        test_normal_count=test_normal,
        exact_overlap_count=exact_overlap,
        exact_overlap_fraction=(
            exact_overlap / test_normal if test_normal else 0.0
        ),
    )


def _blob_digest(blob: object, dataset: str) -> bytes:
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(f"Exp410 SQLite values_blob is invalid: {dataset}")
    return sha256(bytes(blob)).digest()
