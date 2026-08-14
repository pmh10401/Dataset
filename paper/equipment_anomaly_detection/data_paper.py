"""Descriptor-pinned, values-only SQLite loading for paper scoring stages."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from hashlib import sha256
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any

import numpy as np

from .config_paper import FROZEN_CONFIG
from .contracts_paper import FrozenPaperConfig, RawDataset


SqlObserver = Callable[[str], None]
PROGRESS_TICK_INTERVAL_S = 15.0
ALLOWED_PROGRESS_STAGES = frozenset({
    "raw_preload", "task4_task5", "task6", "task7",
    "hard_evidence", "policies", "shard_write", "freeze_verify", "journal_exit",
})
ALLOWED_PROGRESS_PHASES = frozenset({"start", "tick", "end", "error"})
ALLOWED_PROGRESS_KEYS = frozenset({
    "stage", "phase", "completed", "total", "dataset", "detail", "elapsed_s",
})
FORBIDDEN_PROGRESS_SUBSTRINGS = (
    "label=", "p-value", "pvalue=", "score=", "scores=", "rank=", "ranks=",
    "excess_ratio", "alert_index", "alerts=", "select i.label", "split='test' and i.label",
)
ProgressCallback = Callable[[Mapping[str, Any]], None]


def _progress_int(value: object) -> int | None:
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer() and value == value and value not in (float("inf"), float("-inf")):
        return int(value)
    return None


def emit_progress_paper(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    """Best-effort, result-neutral emit. Never raises into science."""
    if callback is None or not isinstance(event, Mapping):
        return
    filtered: dict[str, Any] = {key: event[key] for key in ALLOWED_PROGRESS_KEYS if key in event}
    if filtered.get("stage") not in ALLOWED_PROGRESS_STAGES or filtered.get("phase") not in ALLOWED_PROGRESS_PHASES:
        return
    completed = _progress_int(filtered.get("completed"))
    total = _progress_int(filtered.get("total"))
    if completed is None or total is None:
        return
    filtered["completed"] = completed
    filtered["total"] = total
    elapsed = filtered.get("elapsed_s")
    if type(elapsed) is bool:
        return
    try:
        elapsed_s = float(elapsed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return
    if elapsed_s != elapsed_s or elapsed_s in (float("inf"), float("-inf")) or elapsed_s < 0.0:
        return
    filtered["elapsed_s"] = elapsed_s
    if "dataset" in filtered and type(filtered["dataset"]) is not str:
        filtered.pop("dataset")
    if "detail" in filtered and type(filtered["detail"]) is not str:
        filtered.pop("detail")
    for key in ("dataset", "detail"):
        value = filtered.get(key)
        if type(value) is str and any(token in value.lower() for token in FORBIDDEN_PROGRESS_SUBSTRINGS):
            filtered.pop(key)
    try:
        callback(filtered)
    except BaseException:
        return

TRAIN_VALUES_SQL = """SELECT i.instance_index, i.values_blob
FROM instances i JOIN datasets d ON d.id=i.dataset_id
WHERE d.name=? AND i.split='TRAIN' AND i.label='0' ORDER BY i.instance_index"""
TEST_VALUES_SQL = """SELECT i.instance_index, i.values_blob
FROM instances i JOIN datasets d ON d.id=i.dataset_id
WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"""
POPULATION_SQL = "SELECT name FROM datasets ORDER BY name"
METADATA_LENGTH_SQL = "SELECT series_length FROM datasets WHERE name=?"


def _regular_identity(path: Path) -> tuple[int, int, int, int]:
    details = path.stat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError("database path must be a regular file")
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _descriptor_sha256(descriptor: int) -> str:
    digest = sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


@contextmanager
def _open_verified_read_only(
    db_path: Path,
    *,
    config: FrozenPaperConfig,
) -> Iterator[sqlite3.Connection]:
    """Yield a URI opened through the verified database descriptor only."""
    requested = Path(db_path)
    if requested.is_symlink():
        raise ValueError("database path must not be a symlink")
    supplied = requested.resolve(strict=True)
    expected = Path(config.database_path).resolve(strict=True)
    if supplied != expected:
        raise ValueError("database path differs from frozen authority")
    named_before = _regular_identity(supplied)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(supplied, flags)
    connection: sqlite3.Connection | None = None
    try:
        opened = os.fstat(descriptor)
        opened_identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        if opened_identity != named_before:
            raise ValueError("database identity changed before descriptor pin")
        if _descriptor_sha256(descriptor) != config.database_sha256:
            raise ValueError("database SHA256 differs from frozen authority")
        if _regular_identity(supplied) != opened_identity or os.fstat(descriptor).st_size != opened.st_size:
            raise ValueError("database identity changed while hashing")
        uri = f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only=ON")
        yield connection
        if (
            _regular_identity(supplied) != opened_identity
            or _descriptor_sha256(descriptor) != config.database_sha256
        ):
            raise ValueError("database identity changed during read-only access")
    finally:
        if connection is not None:
            connection.close()
        os.close(descriptor)


def _load_values(
    connection: sqlite3.Connection,
    *,
    dataset: str,
    split: str,
    query: str,
    sql_observer: SqlObserver | None,
) -> tuple[tuple[int, ...], tuple[bytes, ...], tuple[np.ndarray, ...]]:
    if sql_observer is not None:
        sql_observer(query)
    rows = connection.execute(query, (dataset,)).fetchall()
    if not rows:
        raise ValueError(f"no {split} values for dataset={dataset}")
    indices: list[int] = []
    blobs: list[bytes] = []
    values: list[np.ndarray] = []
    for raw_index, raw_blob in rows:
        if type(raw_index) is not int or raw_index < 0:
            raise ValueError(f"{split} instance index is invalid")
        if type(raw_blob) is not bytes or not raw_blob or len(raw_blob) % np.dtype("<f4").itemsize:
            raise ValueError(f"{split} values blob is invalid")
        blob = bytes(raw_blob)
        row = np.frombuffer(blob, dtype="<f4").copy()
        if not len(row):
            raise ValueError(f"{split} values row is empty")
        indices.append(raw_index)
        blobs.append(blob)
        values.append(row)
    if tuple(indices) != tuple(range(len(indices))):
        raise ValueError(f"{split} indices must be contiguous from zero")
    return tuple(indices), tuple(blobs), tuple(values)


def load_population(db_path: Path) -> tuple[str, ...]:
    """Load the frozen, name-sorted population without reading instances."""
    with _open_verified_read_only(db_path, config=FROZEN_CONFIG) as connection:
        rows = connection.execute(POPULATION_SQL).fetchall()
    names = tuple(row[0] for row in rows if row[0] not in FROZEN_CONFIG.excluded_datasets)
    if (
        len(names) != FROZEN_CONFIG.dataset_count
        or tuple(sorted(names)) != names
        or len(set(names)) != len(names)
        or any(type(name) is not str or not name for name in names)
    ):
        raise ValueError("frozen population identity is invalid")
    return names


def load_dataset_values(
    db_path: Path,
    dataset: str,
    *,
    sql_observer: SqlObserver | None = None,
) -> RawDataset:
    """Load normal TRAIN and label-free TEST value rows for one frozen dataset."""
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset name is invalid")
    with _open_verified_read_only(db_path, config=FROZEN_CONFIG) as connection:
        metadata_row = connection.execute(METADATA_LENGTH_SQL, (dataset,)).fetchone()
        if metadata_row is None or type(metadata_row[0]) is not int or metadata_row[0] <= 0:
            raise ValueError("dataset metadata_length must be a positive integer")
        train_indices, train_blobs, train_rows = _load_values(
            connection, dataset=dataset, split="TRAIN", query=TRAIN_VALUES_SQL,
            sql_observer=sql_observer,
        )
        test_indices, test_blobs, test_rows = _load_values(
            connection, dataset=dataset, split="TEST", query=TEST_VALUES_SQL,
            sql_observer=sql_observer,
        )
    return RawDataset(
        dataset=dataset,
        metadata_length=metadata_row[0],
        train_indices=train_indices,
        train_blobs=train_blobs,
        train_rows=train_rows,
        test_indices=test_indices,
        test_blobs=test_blobs,
        test_rows=test_rows,
    )


__all__ = [
    "RawDataset", "TEST_VALUES_SQL", "ProgressCallback", "emit_progress_paper",
    "load_dataset_values", "load_population",
]
