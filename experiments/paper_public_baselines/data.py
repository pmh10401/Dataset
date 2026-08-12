"""Read-only SQLite access that cannot select TEST labels."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sqlite3

import numpy as np

from .config import EXPECTED_DATASETS
from .contracts import DatasetValues


SqlObserver = Callable[[str], None]


def validate_frozen_population(
    dataset_order: tuple[str, ...],
    *,
    expected_count: int = EXPECTED_DATASETS,
) -> tuple[str, ...]:
    """Validate a label-free, publication-pinned dataset order."""
    names = tuple(dataset_order)
    if len(names) != expected_count:
        raise ValueError(
            f"frozen population must contain exactly {expected_count} datasets"
        )
    if any(type(name) is not str or not name for name in names):
        raise ValueError("frozen population contains an invalid dataset name")
    if len(set(names)) != len(names):
        raise ValueError("frozen population must contain unique dataset names")
    return names


def load_frozen_population() -> tuple[str, ...]:
    """Load the exact 1,117-dataset order from the label-free Exp446 publication."""
    from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import (
        capture_exp446_publication_antigravity,
    )

    identity = capture_exp446_publication_antigravity()
    if identity.label_access_count != 0:
        raise ValueError("population authority accessed labels")
    return validate_frozen_population(tuple(identity.dataset_order))


def verify_frozen_population(
    db_path: Path,
    frozen_population: tuple[str, ...],
) -> tuple[str, ...]:
    """Require every frozen dataset in the DB while ignoring later DB additions."""
    frozen = tuple(frozen_population)
    if not frozen or len(frozen) != len(set(frozen)):
        raise ValueError("frozen population must be non-empty and unique")
    available = set(list_datasets_values_only(db_path))
    missing = tuple(name for name in frozen if name not in available)
    if missing:
        raise ValueError(f"frozen population datasets are missing from database: {missing}")
    return frozen


def list_datasets_values_only(db_path: Path) -> tuple[str, ...]:
    """List datasets without touching instance labels or values."""
    connection = _connect_read_only(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM datasets ORDER BY name ASC"
        ).fetchall()
    finally:
        connection.close()
    names = tuple(str(row[0]) for row in rows)
    if not names or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("dataset population is empty or contains duplicates")
    return names


def load_dataset_values(
    db_path: Path,
    dataset: str,
    *,
    sql_observer: SqlObserver | None = None,
) -> DatasetValues:
    """Load only indices and value blobs for one dataset."""
    connection = _connect_read_only(db_path)
    try:
        train_indices, train_rows = _load_split(
            connection, dataset, "TRAIN", sql_observer=sql_observer
        )
        test_indices, test_rows = _load_split(
            connection, dataset, "TEST", sql_observer=sql_observer
        )
    finally:
        connection.close()
    return DatasetValues(
        dataset=str(dataset),
        train_indices=train_indices,
        train_rows=train_rows,
        test_indices=test_indices,
        test_rows=test_rows,
    )


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    resolved = Path(db_path).resolve(strict=True)
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def _load_split(
    connection: sqlite3.Connection,
    dataset: str,
    split: str,
    *,
    sql_observer: SqlObserver | None,
) -> tuple[tuple[int, ...], tuple[np.ndarray, ...]]:
    query = """
        SELECT i.instance_index, i.values_blob
        FROM instances AS i
        JOIN datasets AS d ON d.id = i.dataset_id
        WHERE d.name = ? AND i.split = ?
        ORDER BY i.instance_index ASC
    """
    if sql_observer is not None:
        sql_observer(query)
    rows = connection.execute(query, (str(dataset), str(split))).fetchall()
    if not rows:
        raise ValueError(f"no {split} values for dataset={dataset}")
    indices: list[int] = []
    values: list[np.ndarray] = []
    for raw_index, raw_blob in rows:
        if type(raw_index) is not int or raw_index < 0:
            raise ValueError(f"{split} instance index is invalid")
        if not isinstance(raw_blob, bytes):
            raise ValueError(f"{split} values blob is invalid")
        decoded = np.frombuffer(raw_blob, dtype=np.float32).copy()
        if not len(decoded):
            raise ValueError(f"{split} values row is empty")
        indices.append(raw_index)
        values.append(decoded)
    expected = tuple(range(len(indices)))
    if tuple(indices) != expected:
        raise ValueError(f"{split} indices must be contiguous from zero")
    return tuple(indices), tuple(values)
