"""Values-only prediction input and one-use evaluation-label firewall for Exp451."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import threading
from typing import Callable

import numpy as np

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import _identity_antigravity

from .contracts_antigravity import DatasetBundleAntigravity, FrozenPredictionAuthorityAntigravity
from .predecessor_antigravity import verify_exp448_predecessor_antigravity


_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138
_DATABASE_SIZE_ANTIGRAVITY = 1_753_284_608
_DB_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_VALUE_SQL_ANTIGRAVITY = "SELECT i.split, i.instance_index, i.values_blob FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split IN ('TRAIN','TEST') ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index"
_LABEL_SQL_ANTIGRAVITY = "SELECT i.instance_index, i.label FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"


def _verify_database_antigravity() -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    descriptor = -1
    try:
        before = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        descriptor = os.open(_DATABASE_ANTIGRAVITY, _DB_FLAGS_ANTIGRAVITY)
        opened = os.fstat(descriptor)
        identity = _identity_antigravity(before)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_ino != _DATABASE_INODE_ANTIGRAVITY
                or before.st_size != _DATABASE_SIZE_ANTIGRAVITY or _identity_antigravity(opened) != identity):
            raise ValueError("source database identity differs")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        if digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY or _identity_antigravity(os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)) != identity:
            raise ValueError("source database digest or namespace changed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, identity
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _assert_database_unchanged_antigravity(descriptor: int, identity: tuple[int, int, int, int, int, int, int]) -> None:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1 << 20):
        digest.update(chunk)
    if (digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY or _identity_antigravity(os.fstat(descriptor)) != identity
            or _identity_antigravity(os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)) != identity):
        raise ValueError("source database changed during read")


def _authorizer_antigravity(permitted: frozenset[tuple[str, str]]):
    def authorizer(action: int, table: str | None, column: str | None, _database: str | None, _trigger: str | None) -> int:
        return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and (table, column) in permitted) else sqlite3.SQLITE_DENY
    return authorizer


def _connection_from_descriptor_antigravity(descriptor: int, *, labels: bool):
    connection = sqlite3.connect(f"file:/dev/fd/{descriptor}?mode=ro&immutable=1", uri=True, isolation_level=None)
    permitted = {("instances", "dataset_id"), ("instances", "split"), ("instances", "instance_index"), ("datasets", "id"), ("datasets", "name")}
    permitted.add(("instances", "label" if labels else "values_blob"))
    connection.set_authorizer(_authorizer_antigravity(frozenset(permitted)))
    return connection


def _bundle_from_rows_antigravity(dataset_order: int, family: str, dataset: str, rows: object) -> DatasetBundleAntigravity:
    if type(rows) not in (tuple, list):
        raise ValueError("value rows are invalid")
    train: list[np.ndarray] = []
    test: list[np.ndarray] = []
    expected_split = "TRAIN"
    expected_index = 0
    for row in rows:
        if type(row) not in (tuple, list) or len(row) != 3:
            raise ValueError("value row is invalid")
        split, index, blob = row
        if split == "TEST" and expected_split == "TRAIN" and train:
            expected_split, expected_index = "TEST", 0
        if type(split) is not str or split != expected_split or type(index) is not int or index != expected_index or type(blob) is not bytes or not blob or len(blob) % 4:
            raise ValueError("value row order/index/blob differs")
        values = np.frombuffer(blob, dtype="<f4").astype(np.float64)
        if not np.isfinite(values).all() or values.size == 0 or float(values.std()) == 0.0:
            raise ValueError("value series is nonfinite or zero variance")
        (train if split == "TRAIN" else test).append(values)
        expected_index += 1
    if not train or not test or len({len(item) for item in train + test}) != 1:
        raise ValueError("value coverage or length differs")
    def normalize(items: list[np.ndarray]) -> np.ndarray:
        matrix = np.asarray(items, dtype=np.float64)
        result = (matrix - matrix.mean(axis=1, keepdims=True)) / matrix.std(axis=1, keepdims=True)
        result.setflags(write=False)
        return result
    return DatasetBundleAntigravity(dataset_order, family, dataset, normalize(train), normalize(test), len(test))


def _load_target_bundle_with_connection_antigravity(dataset: str, connection_factory: Callable[[], object]) -> DatasetBundleAntigravity:
    authority = verify_exp448_predecessor_antigravity()
    if type(dataset) is not str or dataset not in authority.target_dataset_names:
        raise ValueError("dataset is not an exact Exp448 target")
    position = authority.target_dataset_names.index(dataset)
    connection = connection_factory()
    try:
        rows = tuple(connection.execute(_VALUE_SQL_ANTIGRAVITY, (dataset,)))
        return _bundle_from_rows_antigravity(authority.target_dataset_order[position], authority.target_baseline_rows[position].family, dataset, rows)
    finally:
        connection.close()


def load_target_bundle_antigravity(dataset: str) -> DatasetBundleAntigravity:
    """Load row-wise z-normalized values without granting label capability."""
    descriptor, identity = _verify_database_antigravity()
    connection = None
    try:
        connection = _connection_from_descriptor_antigravity(descriptor, labels=False)
        result = _load_target_bundle_with_connection_antigravity(dataset, lambda: connection)
        connection = None  # helper closed it exactly once
        _assert_database_unchanged_antigravity(descriptor, identity)
        return result
    finally:
        if connection is not None:
            connection.close()
        os.close(descriptor)


@dataclass(slots=True)
class LabelSessionGuardAntigravity:
    _used_antigravity: bool = False
    _lock_antigravity: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def claim_antigravity(self) -> None:
        with self._lock_antigravity:
            if self._used_antigravity:
                raise ValueError("label session guard is terminal")
            self._used_antigravity = True


@dataclass(frozen=True, slots=True)
class LabelSourceDependenciesAntigravity:
    connection_factory: Callable[[], object]
    prediction_validator: Callable[[FrozenPredictionAuthorityAntigravity], tuple[str, ...]]
    session_guard: LabelSessionGuardAntigravity

    def __post_init__(self) -> None:
        if type(self.session_guard) is not LabelSessionGuardAntigravity or not callable(self.connection_factory) or not callable(self.prediction_validator):
            raise ValueError("label dependencies are invalid")


@dataclass(frozen=True, slots=True)
class LabelSnapshotAntigravity:
    dataset_order: tuple[str, ...]
    labels: tuple[np.ndarray, ...]
    lengths: tuple[int, ...]
    row_count: int
    session_count: int

    def __post_init__(self) -> None:
        snapshots = []
        for value, length in zip(self.labels, self.lengths, strict=True):
            if type(value) is not np.ndarray or value.dtype != np.uint8 or value.ndim != 1 or len(value) != length:
                raise ValueError("label snapshot vector differs")
            copy = np.frombuffer(value.tobytes(), dtype=np.uint8)
            copy.setflags(write=False)
            snapshots.append(copy)
        object.__setattr__(self, "labels", tuple(snapshots))


def _validate_prediction_authority_antigravity(authority: FrozenPredictionAuthorityAntigravity) -> tuple[str, ...]:
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction authority type differs")
    # Canonical round trip rejects semantic aliases and binds names, lengths and indices.
    restored = FrozenPredictionAuthorityAntigravity.from_payload_antigravity(authority.to_payload_antigravity())
    predecessor = verify_exp448_predecessor_antigravity()
    names = tuple(identity[2] for identity in restored.dataset_identities)
    if names != predecessor.target_dataset_names or len(names) != 48 or sum(restored.target_lengths) != 72_300:
        raise ValueError("prediction dataset/length authority differs")
    return names


def _open_ordered_label_session_with_dependencies_antigravity(authority: FrozenPredictionAuthorityAntigravity, dependencies: LabelSourceDependenciesAntigravity) -> LabelSnapshotAntigravity:
    if type(dependencies) is not LabelSourceDependenciesAntigravity:
        raise ValueError("label dependencies type differs")
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction authority type differs")
    FrozenPredictionAuthorityAntigravity.from_payload_antigravity(authority.to_payload_antigravity())
    names = dependencies.prediction_validator(authority)
    if type(names) is not tuple or len(names) != 48 or len(set(names)) != 48 or names != tuple(identity[2] for identity in authority.dataset_identities) or sum(authority.target_lengths) != 72_300:
        raise ValueError("validated prediction order differs")
    dependencies.session_guard.claim_antigravity()
    connection = None
    primary: BaseException | None = None
    try:
        connection = dependencies.connection_factory()
        vectors = []
        for name, expected_length in zip(names, authority.target_lengths, strict=True):
            rows = tuple(connection.execute(_LABEL_SQL_ANTIGRAVITY, (name,)))
            if len(rows) != expected_length:
                raise ValueError("label row count differs")
            labels = np.empty(expected_length, dtype=np.uint8)
            for expected_index, row in enumerate(rows):
                if type(row) not in (tuple, list) or len(row) != 2 or type(row[0]) is not int or row[0] != expected_index or type(row[1]) is not str or row[1] not in ("0", "1"):
                    raise ValueError("label index/text differs")
                labels[expected_index] = int(row[1])
            vectors.append(labels)
        connection.close()
        connection = None
        return LabelSnapshotAntigravity(names, tuple(vectors), tuple(authority.target_lengths), 72_300, 1)
    except BaseException as error:
        primary = error
        raise
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException as cleanup:
                if primary is None:
                    raise cleanup
                raise ExceptionGroup("label session primary and cleanup failures", [primary, cleanup]) from primary


def _production_label_connection_factory_antigravity():
    descriptor, identity = _verify_database_antigravity()
    try:
        connection = _connection_from_descriptor_antigravity(descriptor, labels=True)
    except BaseException:
        os.close(descriptor)
        raise

    class VerifiedLabelConnectionAntigravity:
        __slots__ = ("_closed_antigravity",)

        def __init__(self) -> None:
            self._closed_antigravity = False

        def execute(self, sql: str, parameters: tuple[str]):  # type: ignore[no-untyped-def]
            if self._closed_antigravity:
                raise ValueError("label connection is closed")
            return connection.execute(sql, parameters)

        def close(self) -> None:
            if self._closed_antigravity:
                return
            self._closed_antigravity = True
            primary: BaseException | None = None
            try:
                connection.close()
                _assert_database_unchanged_antigravity(descriptor, identity)
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    os.close(descriptor)
                except BaseException as cleanup:
                    if primary is None:
                        raise
                    raise ExceptionGroup("label database verification and descriptor cleanup failures", [primary, cleanup]) from primary

    return VerifiedLabelConnectionAntigravity()


PRODUCTION_LABEL_CONNECTION_FACTORY_PATH = _production_label_connection_factory_antigravity
_PRODUCTION_LABEL_GUARD_ANTIGRAVITY = LabelSessionGuardAntigravity()


def open_ordered_label_session_antigravity(authority: FrozenPredictionAuthorityAntigravity) -> LabelSnapshotAntigravity:
    """Production-only label opener; tests must use the private dependency seam."""
    dependencies = LabelSourceDependenciesAntigravity(PRODUCTION_LABEL_CONNECTION_FACTORY_PATH, _validate_prediction_authority_antigravity, _PRODUCTION_LABEL_GUARD_ANTIGRAVITY)
    return _open_ordered_label_session_with_dependencies_antigravity(authority, dependencies)


__all__ = ["DatasetBundleAntigravity", "LabelSnapshotAntigravity", "load_target_bundle_antigravity", "open_ordered_label_session_antigravity"]
