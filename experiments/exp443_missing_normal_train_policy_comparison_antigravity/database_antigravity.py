"""Descriptor-bound, label-free SQLite input boundary for Exp443.

The raw database is an immutable input.  A caller first opens and hashes the
published pathname once, then creates a same-inode hard-link in the attempt
directory.  Every worker query is opened from that owned pin and the source
pathname, descriptor and pin are checked on both sides of the query.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import stat
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import RawSeriesAntigravity


NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TRAIN' AND i.label = '0'
ORDER BY i.instance_index"""

TEST_VALUES_SQL_ANTIGRAVITY = """SELECT i.instance_index, i.values_blob
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index"""

TEST_IDENTITY_SQL_ANTIGRAVITY = """SELECT i.instance_index, length(i.values_blob)
FROM instances AS i
JOIN datasets AS d ON d.id = i.dataset_id
WHERE d.name = ? AND i.split = 'TEST'
ORDER BY i.instance_index"""

POPULATION_SQL_ANTIGRAVITY = "SELECT d.name FROM datasets AS d ORDER BY d.name"
QUICK_CHECK_SQL_ANTIGRAVITY = "PRAGMA quick_check"
FROZEN_DATABASE_EXCLUSIONS_ANTIGRAVITY = frozenset({"CornellWhaleChallenge", "Wafer_normal_1"})

# This named seam is intentionally patchable in fixture tests.  Production
# callers use the reviewed value from FROZEN_CONFIG by default.
EXPECTED_DATABASE_SHA256_ANTIGRAVITY = FROZEN_CONFIG.database_sha256

_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_ALLOWED_EXACT_SQL = frozenset({
    NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY,
    TEST_VALUES_SQL_ANTIGRAVITY,
    TEST_IDENTITY_SQL_ANTIGRAVITY,
    POPULATION_SQL_ANTIGRAVITY,
    QUICK_CHECK_SQL_ANTIGRAVITY,
})
_ALLOWED_PRAGMA_SQL = frozenset({"PRAGMA quick_check", "PRAGMA query_only", "PRAGMA query_only=ON"})

# Private deterministic filesystem seams used only by adversarial tests.  The
# hooks perform real rename/FIFO/symlink operations between lstat and open;
# production verification leaves them unset.
_DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY: Callable[[Path], None] | None = None
_AFTER_QUERY_HOOK_ANTIGRAVITY: Callable[[object], None] | None = None


def _metadata(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _hash_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = sha256()
    while block := os.read(descriptor, 1024 * 1024):
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _require_regular(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink < 1:
        raise ValueError(f"{label} must be a linked regular file")
    return info


def _require_directory(path: Path, *, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a regular directory")
    return info


def _descriptor_uri(descriptor: int) -> str:
    root = Path("/dev/fd") if Path("/dev/fd").is_dir() else Path("/proc/self/fd")
    return f"file:{quote(str(root / str(descriptor)), safe='/')}?mode=ro&immutable=1"


def _sql_key(sql: str) -> str:
    if not isinstance(sql, str):
        raise ValueError("SQL must be text")
    return " ".join(sql.strip().split())


def _validate_exact_sql_text_antigravity(sql: str) -> str:
    """Validate a caller SQL string before it reaches sqlite3."""
    key = _sql_key(sql).rstrip(";")
    if key in {_sql_key(item) for item in _ALLOWED_EXACT_SQL} or key in _ALLOWED_PRAGMA_SQL:
        return key
    lowered = key.casefold()
    if any(token in lowered for token in ("labels_blob", "evaluation", "ground_truth", "anomaly")):
        raise ValueError("prediction SQL is label/evaluation capable")
    raise ValueError("unregistered prediction SQL")


def validate_prediction_sql_antigravity(sql: str) -> bool:
    """Allow only the preregistered exact SQL text, before sqlite execution."""
    _validate_exact_sql_text_antigravity(sql)
    return True


def _validate_parameter_contract_antigravity(sql: str, parameters: object) -> str:
    key = _validate_exact_sql_text_antigravity(sql)
    if key in {_sql_key(item) for item in (NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, TEST_VALUES_SQL_ANTIGRAVITY, TEST_IDENTITY_SQL_ANTIGRAVITY)}:
        if type(parameters) not in (tuple, list) or len(parameters) != 1 or type(parameters[0]) is not str or not parameters[0]:
            raise ValueError("dataset SQL requires one non-empty text parameter")
    elif not (type(parameters) in (tuple, list) and len(parameters) == 0):
        raise ValueError("non-query SQL cannot receive parameters")
    return key


def _validate_trace_statement(sql: str) -> None:
    """Trace validation that tolerates SQLite's literal expansion."""
    lowered = _sql_key(sql).casefold().rstrip(";")
    if "labels_blob" in lowered or "evaluation" in lowered or "ground_truth" in lowered or "anomaly" in lowered:
        raise ValueError("forbidden SQL reached SQLite")
    if "split = 'test'" in lowered and "label" in lowered:
        raise ValueError("TEST label predicate reached SQLite")
    if lowered in {_sql_key(item).casefold() for item in _ALLOWED_EXACT_SQL} or lowered in {item.casefold() for item in _ALLOWED_PRAGMA_SQL}:
        return
    if "from instances as i join datasets as d on d.id = i.dataset_id" in lowered:
        if "where d.name =" not in lowered or "order by i.instance_index" not in lowered:
            raise ValueError("unregistered expanded prediction SQL")
        if "i.split = 'train' and i.label = '0'" in lowered and "select i.instance_index, i.values_blob" in lowered:
            return
        if "i.split = 'test'" in lowered and "i.label" not in lowered and ("select i.instance_index, i.values_blob" in lowered or "select i.instance_index, length(i.values_blob)" in lowered):
            return
    if lowered.startswith("pragma quick_check") or lowered.startswith("pragma query_only"):
        return
    raise ValueError("unregistered expanded prediction SQL")


def _make_cursor_facade_antigravity(owner: "PinnedDatabaseAntigravity", cursor: sqlite3.Cursor, connection_facade: object) -> object:
    """Create a zero-state facade whose methods close over the raw cursor."""
    def execute(_self: object, sql: str, parameters: object = ()) -> object:
        return owner._execute_validated(cursor, sql, parameters, connection_facade)

    def fetchone(_self: object) -> tuple[object, ...] | None:
        return cursor.fetchone()

    def fetchmany(_self: object, size: int | None = None) -> list[tuple[object, ...]]:
        return cursor.fetchmany() if size is None else cursor.fetchmany(size)

    def fetchall(_self: object) -> list[tuple[object, ...]]:
        return cursor.fetchall()

    def description(_self: object) -> object:
        return cursor.description

    def rowcount(_self: object) -> int:
        return cursor.rowcount

    def connection(_self: object) -> object:
        return connection_facade

    def iterate(_self: object) -> object:
        return _self

    def next_row(_self: object) -> tuple[object, ...]:
        return next(cursor)

    execute_method = execute
    fetchone_method = fetchone
    fetchmany_method = fetchmany
    fetchall_method = fetchall
    description_property = property(description)
    rowcount_property = property(rowcount)
    connection_property = property(connection)
    iterate_method = iterate
    next_row_method = next_row

    class ValidatedCursorAntigravity:
        __slots__ = ()
        execute = execute_method
        fetchone = fetchone_method
        fetchmany = fetchmany_method
        fetchall = fetchall_method
        description = description_property
        rowcount = rowcount_property
        connection = connection_property
        __iter__ = iterate_method
        __next__ = next_row_method

    return ValidatedCursorAntigravity()


def _make_connection_facade_antigravity(owner: "PinnedDatabaseAntigravity", connection: sqlite3.Connection) -> object:
    """Create a zero-state facade whose methods close over the raw connection."""
    holder: list[object] = []

    def execute(_self: object, sql: str, parameters: object = ()) -> object:
        return owner._execute_validated(connection, sql, parameters, holder[0])

    def cursor(_self: object) -> object:
        return _make_cursor_facade_antigravity(owner, connection.cursor(), holder[0])

    execute_method = execute
    cursor_method = cursor

    class ValidatedConnectionAntigravity:
        __slots__ = ()
        execute = execute_method
        cursor = cursor_method

    facade = ValidatedConnectionAntigravity()
    holder.append(facade)
    return facade


def _decode_rows(rows: Sequence[tuple[object, object]], dataset: str, split: str) -> tuple[RawSeriesAntigravity, ...]:
    if not rows:
        raise ValueError(f"{split} has no rows for {dataset}")
    indices = tuple(index for index, _ in rows)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError(f"{split} instance identity is invalid")
    if indices != tuple(range(len(indices))):
        raise ValueError(f"{split} instance identity is not contiguous from zero")
    decoded: list[RawSeriesAntigravity] = []
    for index, blob in rows:
        if type(blob) is not bytes or not blob or len(blob) % 4:
            raise ValueError(f"{split} values blob is not a non-empty float32 payload")
        values = np.frombuffer(blob, dtype="<f4").copy()
        missing = tuple(int(item) for item in np.flatnonzero(~np.isfinite(values)))
        decoded.append(RawSeriesAntigravity(dataset, split, index, sha256(blob).hexdigest(), values, missing))
    return tuple(decoded)


@dataclass(frozen=True)
class RawDatasetAntigravity:
    dataset: str
    train: tuple[RawSeriesAntigravity, ...]
    test: tuple[RawSeriesAntigravity, ...]
    target_length: int

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("dataset is invalid")
        if not self.train or not self.test:
            raise ValueError("raw dataset population is invalid")
        if type(self.target_length) is not int or not 8 <= self.target_length <= 2048:
            raise ValueError("target_length must be within [8, 2048]")
        if any(item.dataset != self.dataset or item.split != "TRAIN" for item in self.train):
            raise ValueError("TRAIN rows are invalid")
        if any(item.dataset != self.dataset or item.split != "TEST" for item in self.test):
            raise ValueError("TEST rows are invalid")
        expected_target = int(np.clip(np.median([len(item.values) for item in self.train]), 8, 2048))
        if self.target_length != expected_target:
            raise ValueError("target_length must equal clipped original TRAIN median")

    @property
    def train_rows(self) -> tuple[RawSeriesAntigravity, ...]:
        return self.train

    @property
    def test_rows(self) -> tuple[RawSeriesAntigravity, ...]:
        return self.test


@dataclass(frozen=True)
class MissingInventoryAntigravity:
    population: tuple[str, ...]
    affected_datasets: tuple[str, ...]
    affected_families: tuple[str, ...]
    affected_rows: int
    nonfinite_scalars: int
    all_missing_rows: int
    test_rows_by_dataset: Mapping[str, int]
    missing_rows_by_dataset: Mapping[str, int]
    raw_blob_sha256_by_row: Mapping[str, str]
    missing_positions_by_row: Mapping[str, tuple[int, ...]]
    affected_datasets_sha256: str
    label_access_count: int = 0

    def __post_init__(self) -> None:
        population = tuple(self.population)
        if not population or population != tuple(sorted(set(population))):
            raise ValueError("population must be sorted and unique")
        affected = tuple(self.affected_datasets)
        if affected != tuple(sorted(set(affected))) or not set(affected).issubset(population):
            raise ValueError("affected dataset inventory is invalid")
        families = tuple(self.affected_families)
        if families != tuple(sorted(set(families))):
            raise ValueError("affected families are invalid")
        for name in ("affected_rows", "nonfinite_scalars", "all_missing_rows", "label_access_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} is invalid")
        if self.label_access_count != 0:
            raise ValueError("label access is forbidden")
        for value in (self.affected_datasets_sha256,):
            if type(value) is not str or len(value) != 64:
                raise ValueError("inventory SHA256 is invalid")
        rows = {str(k): int(v) for k, v in self.test_rows_by_dataset.items()}
        missing = {str(k): int(v) for k, v in self.missing_rows_by_dataset.items()}
        if set(rows) != set(population) or set(missing) != set(population) or any(v < 1 for v in rows.values()):
            raise ValueError("inventory dataset maps are incomplete")
        raw = {str(k): str(v) for k, v in self.raw_blob_sha256_by_row.items()}
        positions = {str(k): tuple(v) for k, v in self.missing_positions_by_row.items()}
        object.__setattr__(self, "population", population)
        object.__setattr__(self, "affected_datasets", affected)
        object.__setattr__(self, "affected_families", families)
        object.__setattr__(self, "test_rows_by_dataset", MappingProxyType(rows))
        object.__setattr__(self, "missing_rows_by_dataset", MappingProxyType(missing))
        object.__setattr__(self, "raw_blob_sha256_by_row", MappingProxyType(raw))
        object.__setattr__(self, "missing_positions_by_row", MappingProxyType(positions))

    @property
    def datasets(self) -> int:
        return len(self.population)

    @property
    def affected_runs(self) -> int:
        return self.affected_rows

    @property
    def nonfinite_values(self) -> int:
        return self.nonfinite_scalars

    @property
    def all_missing_runs(self) -> int:
        return self.all_missing_rows


class PinnedDatabaseAntigravity:
    """Live owner of the source descriptor and its attempt-local hard link."""

    def __init__(self, database: Path, attempt_root: Path, descriptor: int, pin_path: Path, digest: str, source_metadata: tuple[int, ...], pin_inode: tuple[int, int]):
        self.database_path = Path(database)
        self.attempt_root = Path(attempt_root)
        self.pin_path = Path(pin_path)
        self.database_sha256 = digest
        self.descriptor = descriptor
        self.database_fd = descriptor
        self.source_metadata = source_metadata
        self.pin_inode = pin_inode
        self.parent_descriptor_inode = pin_inode
        self._owned_pin = True
        self._closed = False
        self.sql_trace: list[str] = []
        self.denied_sql_trace: list[str] = []
        self.raw_sql_execute_count = 0
        self.denied_sql_attempts = 0
        self.label_access_count = 0
        self.boundary_trace: list[str] = []
        self._active_sql: str | None = None

    def _require_live(self) -> None:
        if self._closed:
            raise ValueError("pinned database is closed")

    def _verify_boundary(self, boundary: str, *, rehash: bool = False) -> None:
        self._require_live()
        self.boundary_trace.append(boundary)
        current_fd = os.fstat(self.descriptor)
        if _metadata(current_fd) != self.source_metadata:
            raise ValueError("database descriptor metadata changed")
        named = _require_regular(self.database_path, label="database pathname")
        if _metadata(named) != self.source_metadata:
            raise ValueError("database pathname metadata changed")
        pin = _require_regular(self.pin_path, label="owned database pin")
        if _identity(pin) != self.pin_inode or _metadata(pin) != self.source_metadata:
            raise ValueError("owned database pin changed")
        if rehash and _hash_descriptor(self.descriptor) != self.database_sha256:
            raise ValueError("database descriptor SHA256 changed")

    @property
    def pin_inode_metadata(self) -> tuple[int, ...]:
        return _metadata(_require_regular(self.pin_path, label="owned database pin"))

    @property
    def descriptor_inode(self) -> tuple[int, int]:
        return self.pin_inode

    @property
    def database_inode(self) -> tuple[int, int]:
        return self.pin_inode

    def worker_pin_inode_antigravity(self) -> tuple[int, int]:
        self._require_live()
        info = _require_regular(self.pin_path, label="owned database pin")
        if _identity(info) != self.pin_inode:
            raise ValueError("worker pin inode changed")
        return _identity(info)

    def _record_denied_sql(self, sql: str) -> None:
        self.denied_sql_trace.append(sql)
        self.denied_sql_attempts += 1

    def _authorizer(self, action: int, argument_one: str | None, argument_two: str | None, _database: str | None, _trigger: str | None) -> int:
        if action == sqlite3.SQLITE_READ:
            # SQLite authorizer args are (table_name, column_name) for READ.
            table = (argument_one or "").casefold()
            column = (argument_two or "").casefold()
            if table in {"evaluation", "evaluations", "ground_truth", "labels"} or column == "labels_blob":
                return sqlite3.SQLITE_DENY
            approved_train = _sql_key(NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY).casefold()
            if column == "label" and (self._active_sql is None or _sql_key(self._active_sql).casefold() != approved_train):
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def _execute_validated(self, executor: sqlite3.Connection | sqlite3.Cursor, sql: str, parameters: object = (), connection_facade: object | None = None) -> object:
        try:
            normalized = _validate_parameter_contract_antigravity(sql, parameters)
        except ValueError:
            self._record_denied_sql(sql)
            raise
        self.raw_sql_execute_count += 1
        self._active_sql = normalized
        try:
            cursor = executor.execute(sql) if (type(parameters) in (tuple, list) and len(parameters) == 0) else executor.execute(sql, parameters)
        finally:
            self._active_sql = None
        if connection_facade is None:
            raise RuntimeError("validated connection facade is missing")
        return _make_cursor_facade_antigravity(self, cursor, connection_facade)

    @contextmanager
    def worker_connection_antigravity(self) -> Iterator[object]:
        """Open a read-only immutable connection from the retained pin inode."""
        self._verify_boundary("worker:before")
        pin_fd: int | None = None
        connection: sqlite3.Connection | None = None
        try:
            pin_fd = os.open(self.pin_path, _FILE_FLAGS)
            pin_info = os.fstat(pin_fd)
            if _identity(pin_info) != self.pin_inode or _metadata(pin_info) != self.source_metadata:
                raise ValueError("worker connection is not bound to parent pin")
            connection = sqlite3.connect(_descriptor_uri(pin_fd), uri=True)
            connection.set_trace_callback(self._trace)
            connection.set_authorizer(self._authorizer)
            connection.execute("PRAGMA query_only=ON")
            yield _make_connection_facade_antigravity(self, connection)
        except sqlite3.Error as error:
            raise ValueError("immutable worker connection failed") from error
        finally:
            if connection is not None:
                connection.close()
            if pin_fd is not None:
                os.close(pin_fd)
            hook = _AFTER_QUERY_HOOK_ANTIGRAVITY
            if hook is not None:
                hook(self)
            self._verify_boundary("worker:after")

    # Short aliases keep worker call sites explicit while preserving one
    # implementation and one immutable connection policy.
    open_worker_connection_antigravity = worker_connection_antigravity
    worker_connection = worker_connection_antigravity

    def _trace(self, statement: str) -> None:
        self.sql_trace.append(statement)
        try:
            _validate_trace_statement(statement)
        except ValueError:
            # sqlite3 suppresses trace callback exceptions; keep this callback
            # observational and make the execution gate authoritative.
            self.denied_sql_trace.append(statement)

    def close(self, *, terminal_evidence_durable: bool = False) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self._verify_boundary("close:before", rehash=True)
        except BaseException as exc:
            error = exc
        try:
            os.close(self.descriptor)
        except OSError as exc:
            error = error or exc
        if terminal_evidence_durable and self._owned_pin:
            try:
                pin = _require_regular(self.pin_path, label="owned database pin")
                if _identity(pin) != self.pin_inode:
                    raise ValueError("owned database pin changed before cleanup")
                self.pin_path.unlink()
                self._owned_pin = False
            except BaseException as exc:
                error = error or exc
        self._closed = True
        if error is not None:
            raise ValueError("pinned database close failed") from error

    def __enter__(self) -> "PinnedDatabaseAntigravity":
        self._require_live()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close(terminal_evidence_durable=False)


def pin_database_snapshot_antigravity(database: Path | str, attempt_root: Path | str, *, expected_sha256: str | None = None) -> PinnedDatabaseAntigravity:
    """Open, hash, and hard-link a frozen database without changing it."""
    source = Path(database)
    root = Path(attempt_root)
    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
    _require_directory(root, label="attempt root")
    descriptor: int | None = None
    pin_path = root / "database_pin_antigravity.db"
    linked = False
    try:
        # Reject symlinks/FIFOs before calling a blocking read-only open.  In
        # particular, opening a FIFO with O_RDONLY would otherwise wait for a
        # writer and violate the fail-closed input boundary.
        _require_regular(source, label="database pathname")
        hook = _DATABASE_PRE_OPEN_HOOK_ANTIGRAVITY
        if hook is not None:
            hook(source)
        descriptor = os.open(source, _FILE_FLAGS)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise ValueError("database must be a linked regular file")
        digest = _hash_descriptor(descriptor)
        expected = EXPECTED_DATABASE_SHA256_ANTIGRAVITY if expected_sha256 is None else expected_sha256
        if digest != expected:
            raise ValueError("database SHA256 mismatch")
        after_hash = os.fstat(descriptor)
        if _metadata(after_hash) != _metadata(before):
            raise ValueError("database changed while hashing")
        named = _require_regular(source, label="database pathname")
        if _metadata(named) != _metadata(before):
            raise ValueError("database pathname changed while hashing")
        try:
            os.link(source, pin_path, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError("owned database pin already exists") from error
        linked = True
        pin = _require_regular(pin_path, label="database pin")
        if _identity(pin) != _identity(before) or _metadata(os.fstat(descriptor))[:-3] != _metadata(pin)[:-3]:
            raise ValueError("database pin does not bind the hashed inode")
        # A hard link increments nlink in both descriptors.  Re-baseline the
        # complete metadata after linking and retain it for all boundaries.
        operational = os.fstat(descriptor)
        if _identity(operational) != _identity(before) or operational.st_nlink < 2:
            raise ValueError("database pin did not create a same-inode link")
        named_operational = _require_regular(source, label="database pathname")
        if _metadata(named_operational) != _metadata(operational):
            raise ValueError("database pathname changed while pinning")
        pin_operational = _require_regular(pin_path, label="database pin")
        if _metadata(pin_operational) != _metadata(operational):
            raise ValueError("database pin metadata changed while pinning")
        return PinnedDatabaseAntigravity(source, root, descriptor, pin_path, digest, _metadata(operational), _identity(pin_operational))
    except BaseException:
        if linked:
            try:
                pin_path.unlink()
            except OSError:
                pass
        if descriptor is not None:
            os.close(descriptor)
        raise


def load_raw_dataset_antigravity(pinned_database: PinnedDatabaseAntigravity, dataset: str) -> RawDatasetAntigravity:
    """Load only normal TRAIN values and values-only TEST identity."""
    if not isinstance(pinned_database, PinnedDatabaseAntigravity) or type(dataset) is not str or not dataset:
        raise ValueError("raw dataset input is invalid")
    with pinned_database.worker_connection_antigravity() as connection:
        quick = connection.execute(QUICK_CHECK_SQL_ANTIGRAVITY).fetchall()
        if quick != [("ok",)]:
            raise ValueError("SQLite quick_check failed")
        train_rows = connection.execute(NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, (dataset,)).fetchall()
        test_rows = connection.execute(TEST_VALUES_SQL_ANTIGRAVITY, (dataset,)).fetchall()
    train = _decode_rows(train_rows, dataset, "TRAIN")
    test = _decode_rows(test_rows, dataset, "TEST")
    target_length = int(np.clip(np.median([len(item.values) for item in train]), 8, 2048))
    return RawDatasetAntigravity(dataset, train, test, target_length)


def parse_dataset_family_antigravity(dataset: str) -> str:
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset family input is invalid")
    return dataset.rsplit("_normal_", 1)[0] if "_normal_" in dataset else dataset


def scan_missing_inventory_antigravity(
    pinned_database: PinnedDatabaseAntigravity,
    population: Sequence[str],
    *,
    expected_affected_family_count: int | None = FROZEN_CONFIG.affected_family_count,
    expected_test_rows_by_dataset: Mapping[str, int] | None = None,
    expected_target_length_by_dataset: Mapping[str, int] | None = None,
    database_exclusions: frozenset[str] = FROZEN_DATABASE_EXCLUSIONS_ANTIGRAVITY,
) -> MissingInventoryAntigravity:
    """Scan original normal TRAIN blobs and record missingness before repair."""
    if not isinstance(pinned_database, PinnedDatabaseAntigravity):
        raise ValueError("pinned database is invalid")
    names = tuple(population)
    if not names or names != tuple(sorted(set(names))) or any(type(name) is not str or not name for name in names):
        raise ValueError("population must be sorted and unique")
    if expected_affected_family_count is not None and (type(expected_affected_family_count) is not int or expected_affected_family_count < 1):
        raise ValueError("affected family authority is invalid")
    for authority in (expected_test_rows_by_dataset, expected_target_length_by_dataset):
        if authority is not None and set(authority) != set(names):
            raise ValueError("dataset authority population is incomplete")
        if authority is not None and any(type(value) is not int or value < 1 for value in authority.values()):
            raise ValueError("dataset authority values are invalid")
    missing_rows: dict[str, int] = {name: 0 for name in names}
    test_rows: dict[str, int] = {}
    raw_hashes: dict[str, str] = {}
    missing_positions: dict[str, tuple[int, ...]] = {}
    affected: set[str] = set()
    families: set[str] = set()
    affected_row_count = 0
    scalar_count = 0
    all_missing_count = 0
    with pinned_database.worker_connection_antigravity() as connection:
        quick = connection.execute(QUICK_CHECK_SQL_ANTIGRAVITY).fetchall()
        if quick != [("ok",)]:
            raise ValueError("SQLite quick_check failed")
        population_rows = connection.execute(POPULATION_SQL_ANTIGRAVITY).fetchall()
        raw_database_population = tuple(row[0] for row in population_rows)
        if any(len(row) != 1 or type(row[0]) is not str or not row[0] for row in population_rows) or raw_database_population != tuple(sorted(set(raw_database_population))):
            raise ValueError("database population identity is invalid")
        if not isinstance(database_exclusions, frozenset) or any(type(name) is not str or not name for name in database_exclusions) or not database_exclusions.issubset(raw_database_population):
            raise ValueError("database exclusions are invalid")
        database_population = tuple(name for name in raw_database_population if name not in database_exclusions)
        if len(raw_database_population) != len(names) + len(database_exclusions) or database_population != names:
            raise ValueError("database population disagrees with frozen authority")
        for dataset in names:
            rows = connection.execute(NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY, (dataset,)).fetchall()
            decoded = _decode_rows(rows, dataset, "TRAIN")
            if tuple(item.instance_index for item in decoded) != tuple(range(len(decoded))):
                raise ValueError(f"TRAIN instance indices are not contiguous for {dataset}")
            identities = connection.execute(TEST_IDENTITY_SQL_ANTIGRAVITY, (dataset,)).fetchall()
            indices = tuple(index for index, _ in identities)
            if not identities or indices != tuple(range(len(indices))) or len(indices) != len(set(indices)):
                raise ValueError(f"TEST identity coverage is invalid for {dataset}")
            for row in identities:
                if type(row[1]) is not int or row[1] <= 0 or row[1] % 4:
                    raise ValueError(f"TEST identity length is invalid for {dataset}")
            if expected_test_rows_by_dataset is not None and len(identities) != expected_test_rows_by_dataset[dataset]:
                raise ValueError(f"TEST row count disagrees with authority for {dataset}")
            # Exp249 target_length is a transformed-cache metadata authority;
            # raw SQLite blobs may be ragged and are not required to equal the
            # resampled feature length.  The complete keyed authority is
            # validated above and retained in FrozenSourceSnapshot.
            test_rows[dataset] = len(identities)
            for item in decoded:
                key = f"{dataset}/{item.instance_index}"
                raw_hashes[key] = item.raw_blob_sha256
                missing_positions[key] = item.missing_positions
                if item.missing_positions:
                    affected.add(dataset)
                    missing_rows[dataset] += 1
                    affected_row_count += 1
                    scalar_count += len(item.missing_positions)
                    family = parse_dataset_family_antigravity(dataset)
                    families.add(family)
                if not np.isfinite(item.values).any():
                    all_missing_count += 1
    affected_names = tuple(sorted(affected))
    if expected_affected_family_count is not None and len(families) != expected_affected_family_count:
        raise ValueError("affected family count disagrees with frozen authority")
    affected_sha = sha256(json.dumps(list(affected_names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return MissingInventoryAntigravity(
        names, affected_names, tuple(sorted(families)), affected_row_count, scalar_count,
        all_missing_count, test_rows, missing_rows, raw_hashes, missing_positions, affected_sha,
        pinned_database.label_access_count,
    )


__all__ = [
    "NORMAL_TRAIN_VALUES_SQL_ANTIGRAVITY", "TEST_VALUES_SQL_ANTIGRAVITY", "TEST_IDENTITY_SQL_ANTIGRAVITY",
    "POPULATION_SQL_ANTIGRAVITY", "QUICK_CHECK_SQL_ANTIGRAVITY", "FROZEN_DATABASE_EXCLUSIONS_ANTIGRAVITY", "EXPECTED_DATABASE_SHA256_ANTIGRAVITY",
    "PinnedDatabaseAntigravity", "RawDatasetAntigravity", "MissingInventoryAntigravity",
    "pin_database_snapshot_antigravity", "load_raw_dataset_antigravity", "scan_missing_inventory_antigravity",
    "validate_prediction_sql_antigravity", "parse_dataset_family_antigravity",
]
