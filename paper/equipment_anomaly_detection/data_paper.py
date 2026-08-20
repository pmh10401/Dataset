"""Descriptor-pinned, values-only SQLite loading for paper scoring stages."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
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


def _filtered_progress_event_paper(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the one result-neutral event shape shared by best-effort and strict sinks."""
    if not isinstance(event, Mapping):
        return None
    filtered: dict[str, Any] = {key: event[key] for key in ALLOWED_PROGRESS_KEYS if key in event}
    if filtered.get("stage") not in ALLOWED_PROGRESS_STAGES or filtered.get("phase") not in ALLOWED_PROGRESS_PHASES:
        return None
    completed = _progress_int(filtered.get("completed"))
    total = _progress_int(filtered.get("total"))
    if completed is None or total is None:
        return None
    filtered["completed"] = completed
    filtered["total"] = total
    elapsed = filtered.get("elapsed_s")
    if type(elapsed) is bool:
        return None
    try:
        elapsed_s = float(elapsed)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if elapsed_s != elapsed_s or elapsed_s in (float("inf"), float("-inf")) or elapsed_s < 0.0:
        return None
    filtered["elapsed_s"] = elapsed_s
    if "dataset" in filtered and type(filtered["dataset"]) is not str:
        filtered.pop("dataset")
    if "detail" in filtered and type(filtered["detail"]) is not str:
        filtered.pop("detail")
    for key in ("dataset", "detail"):
        value = filtered.get(key)
        if type(value) is str and any(token in value.lower() for token in FORBIDDEN_PROGRESS_SUBSTRINGS):
            filtered.pop(key)
    return filtered


def emit_progress_paper(callback: ProgressCallback | None, event: Mapping[str, Any]) -> None:
    """Best-effort, result-neutral emit. Never raises into paper science."""
    if callback is None:
        return
    filtered = _filtered_progress_event_paper(event)
    if filtered is None:
        return
    try:
        callback(filtered)
    except BaseException:
        return


def emit_strict_progress_paper(callback: ProgressCallback, event: Mapping[str, Any]) -> None:
    """Opt-in operational checkpoint hook; callback durability failures must propagate."""
    if not callable(callback):
        raise TypeError("strict progress callback is required")
    filtered = _filtered_progress_event_paper(event)
    if filtered is None:
        raise ValueError("strict progress event is invalid")
    callback(filtered)

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


def _descriptor_identity(descriptor: int) -> tuple[int, int, int, int]:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode):
        raise ValueError("database descriptor must be a regular file")
    return details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns


def _descriptor_sha256(descriptor: int) -> str:
    digest = sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _frozen_database_path_paper(db_path: Path, config: FrozenPaperConfig) -> Path:
    requested = Path(db_path)
    if requested.is_symlink():
        raise ValueError("database path must not be a symlink")
    supplied = requested.resolve(strict=True)
    expected = Path(config.database_path).resolve(strict=True)
    if supplied != expected:
        raise ValueError("database path differs from frozen authority")
    return supplied


@dataclass(frozen=True, slots=True)
class PreverifiedDatabaseReadOnlyBindingPaper:
    """One full-digest authority bound to an immutable values-only run invocation."""

    database_path: Path
    database_identity: tuple[int, int, int, int]
    database_sha256: str

    def __post_init__(self) -> None:
        path = Path(self.database_path)
        identity = tuple(self.database_identity)
        if (
            not path.is_absolute()
            or len(identity) != 4
            or any(type(value) is not int or value < 0 for value in identity)
            or type(self.database_sha256) is not str
            or len(self.database_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.database_sha256)
        ):
            raise ValueError("preverified database binding is invalid")
        object.__setattr__(self, "database_path", path)
        object.__setattr__(self, "database_identity", identity)


def bind_preverified_database_read_only_paper(
    db_path: Path,
    *,
    config: FrozenPaperConfig | None = None,
    verified_catalog_database: Mapping[str, object] | None = None,
) -> PreverifiedDatabaseReadOnlyBindingPaper:
    """Bind a catalog digest or one fresh digest to one immutable values invocation."""
    authority = FROZEN_CONFIG if config is None else config
    supplied = _frozen_database_path_paper(db_path, authority)
    named_before = _regular_identity(supplied)
    if verified_catalog_database is not None:
        _validated_catalog_database_paper(
            verified_catalog_database,
            supplied=supplied,
            named_identity=named_before,
            config=authority,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(supplied, flags)
    try:
        opened_identity = _descriptor_identity(descriptor)
        if opened_identity != named_before:
            raise ValueError("database identity changed before descriptor pin")
        if (
            verified_catalog_database is None
            and _descriptor_sha256(descriptor) != authority.database_sha256
        ):
            raise ValueError("database SHA256 differs from frozen authority")
        if _regular_identity(supplied) != opened_identity or _descriptor_identity(descriptor) != opened_identity:
            raise ValueError("database identity changed while hashing")
        return PreverifiedDatabaseReadOnlyBindingPaper(
            supplied, opened_identity, authority.database_sha256
        )
    finally:
        os.close(descriptor)


def _validated_preverified_binding_paper(
    binding: PreverifiedDatabaseReadOnlyBindingPaper,
    *,
    supplied: Path,
    config: FrozenPaperConfig,
) -> None:
    if not isinstance(binding, PreverifiedDatabaseReadOnlyBindingPaper):
        raise ValueError("preverified database binding is invalid")
    if binding.database_sha256 != config.database_sha256:
        raise ValueError("preverified database digest differs from frozen authority")
    if binding.database_path != supplied:
        raise ValueError("preverified database path differs from frozen authority")


def _validated_catalog_database_paper(
    database: Mapping[str, object],
    *,
    supplied: Path,
    named_identity: tuple[int, int, int, int],
    config: FrozenPaperConfig,
) -> None:
    expected_keys = {"unresolved_path", "resolved_path", "size", "mtime_ns", "sha256"}
    if (
        not isinstance(database, Mapping)
        or set(database) != expected_keys
        or database["resolved_path"] != str(supplied)
        or type(database["size"]) is not int
        or type(database["mtime_ns"]) is not int
        or database["size"] != named_identity[2]
        or database["mtime_ns"] != named_identity[3]
        or database["sha256"] != config.database_sha256
    ):
        raise ValueError("verified catalog database authority differs")


@contextmanager
def _open_verified_read_only(
    db_path: Path,
    *,
    config: FrozenPaperConfig,
    preverified_database: PreverifiedDatabaseReadOnlyBindingPaper | None = None,
) -> Iterator[sqlite3.Connection]:
    """Yield a URI opened through the verified database descriptor only."""
    supplied = _frozen_database_path_paper(db_path, config)
    if preverified_database is not None:
        _validated_preverified_binding_paper(
            preverified_database, supplied=supplied, config=config
        )
    named_before = _regular_identity(supplied)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(supplied, flags)
    connection: sqlite3.Connection | None = None
    try:
        opened_identity = _descriptor_identity(descriptor)
        if opened_identity != named_before:
            raise ValueError("database identity changed before descriptor pin")
        if (
            preverified_database is not None
            and opened_identity != preverified_database.database_identity
        ):
            raise ValueError("database identity differs from preverified authority")
        if preverified_database is None and _descriptor_sha256(descriptor) != config.database_sha256:
            raise ValueError("database SHA256 differs from frozen authority")
        if _regular_identity(supplied) != opened_identity or _descriptor_identity(descriptor) != opened_identity:
            raise ValueError("database identity changed while hashing")
        uri = f"file:/dev/fd/{descriptor}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None)
        connection.execute("PRAGMA query_only=ON")
        yield connection
        if _regular_identity(supplied) != opened_identity or _descriptor_identity(descriptor) != opened_identity:
            raise ValueError("database identity changed during read-only access")
        if preverified_database is None and _descriptor_sha256(descriptor) != config.database_sha256:
            raise ValueError("database SHA256 differs from frozen authority")
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
    preverified_database: PreverifiedDatabaseReadOnlyBindingPaper | None = None,
) -> RawDataset:
    """Load normal TRAIN and label-free TEST value rows for one frozen dataset."""
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset name is invalid")
    with _open_verified_read_only(
        db_path, config=FROZEN_CONFIG, preverified_database=preverified_database
    ) as connection:
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
    "PreverifiedDatabaseReadOnlyBindingPaper", "RawDataset", "TEST_VALUES_SQL", "ProgressCallback",
    "bind_preverified_database_read_only_paper", "emit_progress_paper", "emit_strict_progress_paper",
    "load_dataset_values", "load_population",
]
