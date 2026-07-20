from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.exp249_250_official_mini_multirocket.local_gap import LocalGapModel
from .config import CROSSFIT_FOLDS, KNN_NEIGHBORS


@dataclass(frozen=True)
class FeatureScores:
    train_oof: np.ndarray
    test_scores: np.ndarray


def crossfit_plan(n_train: int, *, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if n_train < 2:
        raise ValueError("at least two TRAIN rows are required")
    indices = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(indices)
    queries = tuple(
        np.sort(part)
        for part in np.array_split(indices, min(CROSSFIT_FOLDS, n_train))
        if len(part)
    )
    return tuple(
        (
            np.sort(np.concatenate([part for other, part in enumerate(queries) if other != index])),
            query,
        )
        for index, query in enumerate(queries)
    )


def score_feature_blocks(
    train_blocks: dict[str, np.ndarray], test_blocks: dict[str, np.ndarray], *, seed: int
) -> dict[str, FeatureScores]:
    if set(train_blocks) != set(test_blocks):
        raise ValueError("TRAIN and TEST feature block sets must match")
    output = {}
    for name in train_blocks:
        train = np.asarray(train_blocks[name], dtype=float)
        test = np.asarray(test_blocks[name], dtype=float)
        if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
            raise ValueError(f"invalid feature matrices: block={name}")
        train_oof = np.empty(len(train), dtype=float)
        for fit_indices, query_indices in crossfit_plan(len(train), seed=seed):
            train_oof[query_indices] = LocalGapModel(
                train[fit_indices], neighbors=KNN_NEIGHBORS
            ).score(train[query_indices])
        test_scores = LocalGapModel(train, neighbors=KNN_NEIGHBORS).score(test)
        if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
            raise ValueError(f"non-finite local-gap scores: block={name}")
        output[name] = FeatureScores(train_oof=train_oof, test_scores=test_scores)
    return output

