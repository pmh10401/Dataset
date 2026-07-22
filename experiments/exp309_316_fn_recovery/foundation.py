from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle, preprocessing_contract
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle

from .config import DB_PATH, SEED


EXPECTED_PREPROCESSING_ORDER = [
    "missing_repair", "train_length_alignment", "normalization",
]


def _z_normalize_per_run(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return ((values - center) / np.where(scale > 1e-8, scale, 1.0)).astype(np.float32)


def load_preprocessed_bundle(dataset: str, path: Path = DB_PATH) -> DatasetBundle:
    contract = preprocessing_contract()
    if contract.get("order") != EXPECTED_PREPROCESSING_ORDER:
        raise ValueError(f"preprocessing contract mismatch: {contract}")
    aligned = align_bundle(load_values_bundle(str(dataset), Path(path)))
    train = _z_normalize_per_run(np.asarray(aligned.train_normal, dtype=np.float32))
    test = _z_normalize_per_run(np.asarray(aligned.test_values, dtype=np.float32))
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(f"non-finite preprocessed values: dataset={dataset}")
    return DatasetBundle(str(dataset), train, test, int(aligned.target_length))


def deterministic_folds(n_train: int, *, seed: int = SEED, folds: int = 5) -> tuple[np.ndarray, ...]:
    if int(n_train) < 2:
        raise ValueError("at least two TRAIN rows are required")
    indices = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(indices)
    return tuple(np.sort(part) for part in np.array_split(indices, min(int(folds), int(n_train))) if len(part))


def crossfit_plan(n_train: int, *, seed: int = SEED, folds: int = 5) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    parts = deterministic_folds(n_train, seed=seed, folds=folds)
    plan = []
    for index, query in enumerate(parts):
        fit_parts = [part for other, part in enumerate(parts) if other != index]
        if not fit_parts:
            raise ValueError("cross-fit requires at least two folds")
        plan.append((np.sort(np.concatenate(fit_parts)), query))
    return tuple(plan)


def list_datasets_small_to_large(path: Path = DB_PATH, limit: int | None = None) -> list[str]:
    resolved = Path(path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT d.name, COUNT(i.id) AS n_test
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id AND i.split = 'TEST'
            GROUP BY d.id, d.name
            ORDER BY n_test ASC, d.name ASC
            """
        ).fetchall()
    finally:
        connection.close()
    names = [str(row[0]) for row in rows]
    return names[: int(limit)] if limit is not None else names
