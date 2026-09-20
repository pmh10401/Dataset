"""Read only the frozen ECG5000 normal_2 TRAIN matrix for Exp459."""

from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_EXPECTED_TRAIN_ROWS_ANTIGRAVITY = 1413
_EXPECTED_WIDTH_ANTIGRAVITY = 140
_TRAIN_SQL_ANTIGRAVITY = (
    "SELECT i.instance_index, i.values_blob "
    "FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id "
    "WHERE d.name = ? AND i.split = 'TRAIN' "
    "ORDER BY i.instance_index"
)


def _connection_antigravity() -> sqlite3.Connection:
    resolved = _DATABASE_ANTIGRAVITY.resolve(strict=True)
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def load_normal2_train_antigravity() -> np.ndarray:
    """Return the exact owned read-only float32 normal_2 TRAIN matrix."""
    connection = _connection_antigravity()
    try:
        rows = tuple(connection.execute(_TRAIN_SQL_ANTIGRAVITY, (FROZEN_CONFIG_ANTIGRAVITY.dataset,)))
    finally:
        connection.close()

    if len(rows) != _EXPECTED_TRAIN_ROWS_ANTIGRAVITY:
        raise ValueError("TRAIN row count does not match the frozen normal_2 contract")
    if tuple(row[0] for row in rows) != tuple(range(_EXPECTED_TRAIN_ROWS_ANTIGRAVITY)):
        raise ValueError("TRAIN indices must be exact, contiguous, unique, and ordered")

    decoded: list[np.ndarray] = []
    for _, blob in rows:
        if type(blob) is not bytes or not blob or len(blob) % np.dtype("<f4").itemsize:
            raise ValueError("TRAIN values_blob must contain non-empty float32 bytes")
        decoded.append(np.frombuffer(blob, dtype="<f4"))
    if not decoded:
        raise ValueError("TRAIN rows must exist")
    if len({row.size for row in decoded}) != 1 or decoded[0].size != _EXPECTED_WIDTH_ANTIGRAVITY:
        raise ValueError("TRAIN shape must be the frozen 1413x140 matrix")

    result = np.array(np.stack(decoded), dtype=np.float32, copy=True, order="C")
    if result.shape != (_EXPECTED_TRAIN_ROWS_ANTIGRAVITY, _EXPECTED_WIDTH_ANTIGRAVITY):
        raise ValueError("TRAIN shape must be the frozen 1413x140 matrix")
    if not np.isfinite(result).all():
        raise ValueError("TRAIN rows must be finite")
    result.setflags(write=False)
    return result


__all__ = ("load_normal2_train_antigravity",)
