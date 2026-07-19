from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import (
    load_values_bundle,
    sha256_file,
)
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp239_248_diverse_score_sources.config import DB_PATH
from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle


def deterministic_folds(
    n_train: int, *, seed: int, folds: int = 5
) -> tuple[np.ndarray, ...]:
    if n_train < 2:
        raise ValueError("at least two TRAIN rows are required")
    count = min(int(folds), int(n_train))
    shuffled = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    return tuple(np.sort(part) for part in np.array_split(shuffled, count) if len(part))


def crossfit_plan(
    n_train: int, *, seed: int, folds: int = 5
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    queries = deterministic_folds(n_train, seed=seed, folds=folds)
    plan = []
    for index, query in enumerate(queries):
        fit = np.concatenate([part for other, part in enumerate(queries) if other != index])
        plan.append((np.sort(fit), query))
    return tuple(plan)


def _z_normalize(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return ((values - center) / np.where(scale > 1e-8, scale, 1.0)).astype(np.float32)


def load_bundle(dataset: str, path: Path = DB_PATH) -> DatasetBundle:
    raw = align_bundle(load_values_bundle(str(dataset), path))
    train = _z_normalize(np.asarray(raw.train_normal, dtype=np.float32))
    test = _z_normalize(np.asarray(raw.test_values, dtype=np.float32))
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(f"non-finite aligned values: dataset={dataset}")
    return DatasetBundle(
        dataset=str(dataset),
        train_normal=train,
        test_values=test,
        target_length=int(raw.target_length),
    )


def stable_json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def database_hash(path: Path = DB_PATH) -> str:
    return sha256_file(path)

