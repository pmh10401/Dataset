from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp249_250_official_mini_multirocket.config import (
    CROSSFIT_FOLDS,
    KNN_NEIGHBORS,
    MAX_TRAIN_REFERENCE,
    QUERY_CHUNK_SIZE,
    RocketModelConfig,
)
from experiments.exp249_250_official_mini_multirocket.local_gap import LocalGapModel
from experiments.exp249_250_official_mini_multirocket.transforms import (
    as_aeon_collection,
    make_transformer,
)


@dataclass(frozen=True)
class ConfigurationScores:
    train_oof: np.ndarray
    test_scores: np.ndarray
    feature_dimension: int


def _features(
    transformer, fit_rows: np.ndarray, query_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fit = np.asarray(transformer.fit_transform(as_aeon_collection(fit_rows)), dtype=float)
    query = np.asarray(transformer.transform(as_aeon_collection(query_rows)), dtype=float)
    if fit.ndim != 2 or query.ndim != 2:
        raise ValueError("official ROCKET features must be two-dimensional")
    return fit, query


def _reference_indices(indices: np.ndarray, *, maximum: int, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= int(maximum):
        return values
    selected = np.random.default_rng(int(seed)).choice(
        values, size=int(maximum), replace=False
    )
    return np.sort(selected)


def _score_query_chunks(
    transformer,
    model: LocalGapModel,
    rows: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(rows), int(chunk_size)):
        features = np.asarray(
            transformer.transform(as_aeon_collection(rows[start : start + chunk_size])),
            dtype=float,
        )
        parts.append(model.score(features))
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def score_configuration(
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    *,
    config: RocketModelConfig,
    requested_kernels: int,
    seed: int,
    expected_dimension: int | None,
    max_train_reference: int = MAX_TRAIN_REFERENCE,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
) -> ConfigurationScores:
    train = np.asarray(train_rows, dtype=np.float32)
    test = np.asarray(test_rows, dtype=np.float32)
    train_oof = np.empty(len(train), dtype=float)
    dimensions: set[int] = set()
    for fold_index, (fit_indices, query_indices) in enumerate(crossfit_plan(
        len(train), seed=int(seed), folds=CROSSFIT_FOLDS
    )):
        reference_indices = _reference_indices(
            fit_indices,
            maximum=max_train_reference,
            seed=int(seed) + 1009 * (fold_index + 1) + int(requested_kernels),
        )
        transformer = make_transformer(
            config, requested_kernels=requested_kernels, seed=seed
        )
        fit_features = np.asarray(
            transformer.fit_transform(as_aeon_collection(train[reference_indices])),
            dtype=float,
        )
        dimensions.add(int(fit_features.shape[1]))
        model = LocalGapModel(fit_features, neighbors=KNN_NEIGHBORS)
        train_oof[query_indices] = _score_query_chunks(
            transformer,
            model,
            train[query_indices],
            chunk_size=query_chunk_size,
        )

    full_reference_indices = _reference_indices(
        np.arange(len(train), dtype=int),
        maximum=max_train_reference,
        seed=int(seed) + 7919 + int(requested_kernels),
    )
    transformer = make_transformer(
        config, requested_kernels=requested_kernels, seed=seed
    )
    train_features = np.asarray(
        transformer.fit_transform(as_aeon_collection(train[full_reference_indices])),
        dtype=float,
    )
    dimensions.add(int(train_features.shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    feature_dimension = dimensions.pop()
    if expected_dimension is not None and feature_dimension != int(expected_dimension):
        raise ValueError(
            "feature dimension mismatch: "
            f"expected={expected_dimension} actual={feature_dimension}"
        )
    test_scores = _score_query_chunks(
        transformer,
        LocalGapModel(train_features, neighbors=KNN_NEIGHBORS),
        test,
        chunk_size=query_chunk_size,
    )
    if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
        raise ValueError("official ROCKET produced non-finite local-gap scores")
    return ConfigurationScores(train_oof, test_scores, feature_dimension)
