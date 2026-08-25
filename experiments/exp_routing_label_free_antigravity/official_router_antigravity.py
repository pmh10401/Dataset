"""Official routing entry point: label-free method selection for any dataset.

This module is the single integration surface the official pipeline should
call. It never reads labels; only TRAIN majority rows and their features.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np

from .feature_extractor_antigravity import extract_routing_features_antigravity
from .routing_rules_antigravity import route_dataset_antigravity

_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_TRAIN_SAMPLE_CAP_ANTIGRAVITY = 300

_VALUE_SQL_ANTIGRAVITY = (
    "SELECT i.split, i.values_blob FROM instances AS i "
    "JOIN datasets AS d ON d.id=i.dataset_id "
    "WHERE d.name=? AND i.label='0' AND i.split='TRAIN' ORDER BY i.instance_index"
)


def load_train_features_antigravity(
    dataset: str, *, connection: sqlite3.Connection | None = None
) -> dict[str, float]:
    """Return label-free features for a dataset from its TRAIN label=0 rows."""
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
    blobs = [row[1] for row in rows]
    if len(blobs) > _TRAIN_SAMPLE_CAP_ANTIGRAVITY:
        rng = np.random.default_rng(20260825)
        index = np.sort(rng.choice(len(blobs), size=_TRAIN_SAMPLE_CAP_ANTIGRAVITY, replace=False))
        blobs = [blobs[i] for i in index]
    matrix = np.stack([np.frombuffer(blob, dtype="<f4") for blob in blobs]).astype(np.float64, order="C")
    return extract_routing_features_antigravity(matrix)


def route_official_antigravity(dataset: str) -> str:
    """Public API: return the recommended method for a dataset by name alone."""
    features = load_train_features_antigravity(dataset)
    return route_dataset_antigravity(features)


__all__ = ["load_train_features_antigravity", "route_official_antigravity"]
