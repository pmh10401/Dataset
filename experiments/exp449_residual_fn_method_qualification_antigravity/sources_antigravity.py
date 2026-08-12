"""Read-only value source with a hard boundary before the Task 6 label session."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import stat
from typing import Callable

import numpy as np

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import _identity_antigravity

from .contracts_antigravity import DatasetBundleAntigravity


_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_DATABASE_SHA256_ANTIGRAVITY = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
_DATABASE_INODE_ANTIGRAVITY = 1048138
_DB_FLAGS_ANTIGRAVITY = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_VALUE_SQL_ANTIGRAVITY = "SELECT i.split, i.instance_index, i.values_blob FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split IN ('TRAIN','TEST') ORDER BY CASE i.split WHEN 'TRAIN' THEN 0 ELSE 1 END, i.instance_index"
_LABEL_SQL_ANTIGRAVITY = "SELECT i.instance_index, i.label FROM instances AS i JOIN datasets AS d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"


def _verify_database_antigravity() -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    try:
        before = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        descriptor = os.open(_DATABASE_ANTIGRAVITY, _DB_FLAGS_ANTIGRAVITY)
    except OSError as error:
        raise ValueError("source database is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        identity = _identity_antigravity(before)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_ino != _DATABASE_INODE_ANTIGRAVITY or _identity_antigravity(opened) != identity:
            raise ValueError("source database inode/link contract is invalid")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
        if digest.hexdigest() != _DATABASE_SHA256_ANTIGRAVITY or _identity_antigravity(current) != identity:
            raise ValueError("source database changed during digest verification")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _connection_antigravity(descriptor: int):
    connection = sqlite3.connect(f"file:/dev/fd/{descriptor}?mode=ro&immutable=1", uri=True, isolation_level=None)
    permitted = frozenset({
        ("instances", "dataset_id"), ("instances", "split"), ("instances", "instance_index"),
        ("instances", "values_blob"), ("instances", "label"), ("datasets", "id"), ("datasets", "name"),
    })
    def authorizer(action: int, table: str | None, column: str | None, _database: str | None, _trigger: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT or (action == sqlite3.SQLITE_READ and (table, column) in permitted):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    connection.set_authorizer(authorizer)
    return connection


def _normalise_antigravity(rows: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("source value matrix is invalid")
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError("source series has zero variance")
    return (values - mean) / scale


def load_target_bundle_antigravity(dataset: str, connection_factory: Callable[[int], object] | None = None) -> DatasetBundleAntigravity:
    """Load only split, index, and value bytes; this function has no label capability."""
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset is invalid")
    descriptor = -1
    connection = None
    try:
        if connection_factory is None:
            descriptor, identity = _verify_database_antigravity()
            connection = _connection_antigravity(descriptor)
        else:
            connection = connection_factory(-1)
            identity = None
        rows = tuple(connection.execute(_VALUE_SQL_ANTIGRAVITY, (dataset,)))
        train: list[np.ndarray] = []
        test: list[np.ndarray] = []
        expected = {"TRAIN": 0, "TEST": 0}
        for split, index, blob in rows:
            if split not in expected or type(index) is not int or index != expected[split] or type(blob) is not bytes or not blob or len(blob) % 4:
                raise ValueError("source value rows are invalid")
            expected[split] += 1
            (train if split == "TRAIN" else test).append(np.frombuffer(blob, dtype="<f4"))
        if not train or not test or len({item.size for item in train + test}) != 1:
            raise ValueError("source value coverage is invalid")
        result = DatasetBundleAntigravity(dataset, _normalise_antigravity(train), _normalise_antigravity(test), len(test))
        connection.close()
        connection = None
        if descriptor >= 0:
            opened = os.fstat(descriptor)
            current = os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
            if _identity_antigravity(opened) != identity or _identity_antigravity(current) != identity or opened.st_nlink != 1:
                raise ValueError("source database changed during value load")
        return result
    finally:
        if connection is not None:
            connection.close()
        if descriptor >= 0:
            os.close(descriptor)


def convert_test_label_row_antigravity(instance_index: object, raw_label: object) -> tuple[int, int]:
    if type(instance_index) is not int or type(raw_label) is not str or raw_label not in ("0", "1"):
        raise ValueError("TEST label row is invalid")
    return instance_index, 0 if raw_label == "0" else 1


def open_test_labels_once_antigravity(prediction_authority: object, *, connection_factory: Callable[[int], object] | None = None):
    """Task 6-only closure session; no Task 3 caller can manufacture its authority."""
    if type(prediction_authority).__name__ != "FrozenPredictionAuthorityAntigravity" or type(prediction_authority).__module__ != "experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity":
        raise ValueError("prediction authority is invalid")
    required = ("family_method_pairs", "dataset_units", "label_access_count")
    if tuple(getattr(prediction_authority, name, None) for name in required) != (16, 96, 0):
        raise ValueError("prediction authority is invalid")
    descriptor, identity = _verify_database_antigravity()
    connection = None
    try:
        connection = _connection_antigravity(descriptor) if connection_factory is None else connection_factory(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    closed = False
    close_failure: BaseException | None = None
    descriptor_released = False
    next_dataset = 0
    order: tuple[str, ...] = getattr(prediction_authority, "target_dataset_order", ())

    def release_descriptor_antigravity() -> None:
        nonlocal descriptor_released
        if not descriptor_released:
            os.close(descriptor)
            descriptor_released = True

    class LabelSessionAntigravity:
        __slots__ = ()
        def close_state_antigravity(self) -> str:
            return "closed" if closed else ("close_failed" if close_failure is not None else "open")
        def close_error_antigravity(self) -> BaseException | None:
            return close_failure
        def labels_for_dataset_antigravity(self, dataset: str) -> tuple[tuple[int, int], ...]:
            nonlocal next_dataset
            if closed or next_dataset >= len(order) or dataset != order[next_dataset]:
                raise ValueError("TEST labels must be read once in frozen dataset order")
            rows = tuple(convert_test_label_row_antigravity(index, value) for index, value in connection.execute(_LABEL_SQL_ANTIGRAVITY, (dataset,)))
            if tuple(index for index, _ in rows) != tuple(range(len(rows))):
                raise ValueError("TEST indices are not contiguous")
            next_dataset += 1
            return rows
        def close_antigravity(self) -> None:
            nonlocal closed, close_failure
            if closed:
                return
            if close_failure is not None:
                raise close_failure
            try:
                if next_dataset != len(order):
                    raise ValueError("TEST label session did not complete the frozen dataset order")
                connection.close()
                opened, current = os.fstat(descriptor), os.stat(_DATABASE_ANTIGRAVITY, follow_symlinks=False)
                if _identity_antigravity(opened) != identity or _identity_antigravity(current) != identity or opened.st_nlink != 1:
                    raise ValueError("source database changed during TEST-label session")
                release_descriptor_antigravity()
                closed = True
            except BaseException as error:
                close_failure = error
                try:
                    release_descriptor_antigravity()
                except OSError:
                    pass
                raise
    return LabelSessionAntigravity()


__all__ = ["convert_test_label_row_antigravity", "load_target_bundle_antigravity", "open_test_labels_once_antigravity"]
