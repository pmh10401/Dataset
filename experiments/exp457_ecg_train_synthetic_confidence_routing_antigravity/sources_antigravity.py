"""Read only the five frozen ECG5000 TRAIN value matrices."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_EXPECTED_TRAIN_ROWS_ANTIGRAVITY = MappingProxyType(
    dict(zip(FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets, (2335, 1413, 76, 155, 19), strict=True))
)
_TRAIN_SQL_ANTIGRAVITY = (
    "SELECT i.instance_index, i.values_blob "
    "FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id "
    "WHERE d.name = ? AND i.split = 'TRAIN' "
    "ORDER BY i.instance_index"
)


def _connection_antigravity() -> sqlite3.Connection:
    resolved = _DATABASE_ANTIGRAVITY.resolve(strict=True)
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def load_ecg_train_antigravity(dataset: str) -> np.ndarray:
    """Return one exact-index, owned, read-only float32 TRAIN matrix."""
    if type(dataset) is not str or dataset not in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("dataset must be one of the five exact ECG5000 datasets")

    connection = _connection_antigravity()
    try:
        rows = tuple(connection.execute(_TRAIN_SQL_ANTIGRAVITY, (dataset,)))
    finally:
        connection.close()

    expected_count = _EXPECTED_TRAIN_ROWS_ANTIGRAVITY[dataset]
    if len(rows) != expected_count:
        raise ValueError("TRAIN indices do not have the frozen row count")
    if tuple(row[0] for row in rows) != tuple(range(expected_count)):
        raise ValueError("TRAIN indices must be exact, contiguous, unique, and ordered")

    decoded: list[np.ndarray] = []
    for _, blob in rows:
        if type(blob) is not bytes or not blob or len(blob) % np.dtype("<f4").itemsize:
            raise ValueError("TRAIN values_blob must contain non-empty float32 bytes")
        decoded.append(np.frombuffer(blob, dtype="<f4"))
    if not decoded or len({row.size for row in decoded}) != 1:
        raise ValueError("TRAIN rows must have one common length")

    result = np.array(np.stack(decoded), dtype=np.float32, copy=True, order="C")
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("TRAIN rows must be a finite two-dimensional float32 matrix")
    result.setflags(write=False)
    return result


__all__ = ("load_ecg_train_antigravity",)
