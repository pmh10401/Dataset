"""Official routing entry point: label-free method selection for any dataset.

This module is the single integration surface the official pipeline should
call. It reads only normal TRAIN rows and passes numeric features to the
qualified Exp476 policy; the dataset name is never a policy feature.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
import numpy as np

from experiments.exp474_conservative_fallback_routing_antigravity.ragged_features_antigravity import (
    extract_ragged_routing_features_antigravity,
)

from .conservative_policy_antigravity import route_features_official_antigravity

_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_VALUE_SQL_ANTIGRAVITY = (
    "SELECT i.values_blob FROM instances AS i "
    "JOIN datasets AS d ON d.id=i.dataset_id "
    "WHERE d.name=? AND i.label='0' AND i.split='TRAIN' ORDER BY i.instance_index"
)


def load_train_features_antigravity(
    dataset: str, *, connection: sqlite3.Connection | None = None
) -> dict[str, float]:
    """Return the seven Exp476 features from every normal TRAIN row."""
    own_connection = connection is None
    if own_connection:
        connection = sqlite3.connect(f"file:{_DATABASE_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        rows = connection.execute(_VALUE_SQL_ANTIGRAVITY, (dataset,)).fetchall()
    finally:
        if own_connection and connection is not None:
            connection.close()
    if not rows:
        raise ValueError(f"dataset {dataset!r} has no TRAIN label=0 rows")
    train_rows = [
        np.frombuffer(row[0], dtype="<f4").astype(np.float64, copy=False) for row in rows
    ]
    finite_train_rows = [row for row in train_rows if np.isfinite(row).all()]
    if len(finite_train_rows) < 2:
        raise ValueError(
            f"dataset {dataset!r} has fewer than two finite TRAIN label=0 rows"
        )
    return extract_ragged_routing_features_antigravity(finite_train_rows)


def route_official_antigravity(dataset: str) -> str:
    """Load a dataset's TRAIN rows and return its Exp476-qualified method."""
    features = load_train_features_antigravity(dataset)
    return route_features_official_antigravity(features)


__all__ = ["load_train_features_antigravity", "route_official_antigravity"]
