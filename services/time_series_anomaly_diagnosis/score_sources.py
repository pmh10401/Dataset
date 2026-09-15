"""Service-local implementations of the three approved anomaly score sources."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aeon.transformations.collection.convolution_based import (
    HydraTransformer,
    MiniRocket,
    MultiRocket,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


METHOD_NAMES = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
)
_REQUESTED_KERNELS = dict(zip(METHOD_NAMES, (4096, 1024, 8), strict=True))
_CROSSFIT_FOLDS = 5
_MAX_TRAIN_REFERENCE = 1000
_QUERY_CHUNK_SIZE = 128


def _protected_float_array(values: np.ndarray, field: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{field} must be a finite one-dimensional ndarray")
    return np.frombuffer(values.astype(float, copy=False).tobytes(), dtype=float)


@dataclass(frozen=True, slots=True)
class ScoreResult:
    train_oof: np.ndarray
    test_scores: np.ndarray
    feature_dimension: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.feature_dimension) is not int or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be an exact positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be an exact non-negative integer")
        object.__setattr__(self, "train_oof", _protected_float_array(self.train_oof, "train_oof"))
        object.__setattr__(self, "test_scores", _protected_float_array(self.test_scores, "test_scores"))


def crossfit_plan(n_train: int, *, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if type(n_train) is not int or n_train < _CROSSFIT_FOLDS:
        raise ValueError("at least five TRAIN rows are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    shuffled = np.arange(n_train, dtype=int)
    np.random.default_rng(seed).shuffle(shuffled)
    queries = tuple(np.sort(part) for part in np.array_split(shuffled, _CROSSFIT_FOLDS))
    return tuple(
        (
            np.sort(np.concatenate([part for other, part in enumerate(queries) if other != index])),
            query,
        )
        for index, query in enumerate(queries)
    )


def _rows(rows: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if (
        values.ndim != 2
        or len(values) == 0
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"{field} must be a non-empty finite two-dimensional array")
    return values


def _collection(rows: np.ndarray) -> np.ndarray:
    values = _rows(rows, "rows")
    if values.shape[1] < 9:
        values = np.pad(values, ((0, 0), (0, 9 - values.shape[1])))
    return values[:, np.newaxis, :]


def _features(values: object, *, expected_dimension: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("non-finite representation features")
    if expected_dimension is not None and result.shape[1] != expected_dimension:
        raise ValueError("inconsistent feature dimensions")
    return result


class _Representation:
    def __init__(self, transformer: object) -> None:
        self._transformer = transformer

    def fit_transform(self, rows: np.ndarray) -> np.ndarray:
        return _features(self._transformer.fit_transform(_collection(rows)))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return _features(self._transformer.transform(_collection(rows)))


def _make_representation(method_name: str, seed: int) -> _Representation:
    if method_name == "minirocket_local_gap":
        transformer = MiniRocket(n_kernels=4096, n_jobs=1, random_state=seed)
    elif method_name == "multirocket_local_gap":
        transformer = MultiRocket(
            n_kernels=1024,
            max_dilations_per_kernel=32,
            n_features_per_kernel=4,
            normalise=False,
            n_jobs=1,
            random_state=seed,
        )
    elif method_name == "hydra_local_gap":
        transformer = HydraTransformer(
            n_kernels=8,
            n_groups=64,
            n_jobs=1,
            random_state=seed,
        )
    else:
        raise ValueError("method_name must be one of the three approved score sources")
    return _Representation(transformer)


class _LocalGapModel:
    def __init__(self, reference: np.ndarray) -> None:
        values = np.asarray(reference, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("local-gap requires a two-dimensional reference")
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(values)
        scaled = self.scaler.transform(values)
        self.neighbors = max(1, min(3, len(scaled)))
        self.reference_model = NearestNeighbors(
            n_neighbors=max(1, min(self.neighbors + 1, len(scaled))),
            metric="euclidean",
        ).fit(scaled)
        distances, indices = self.reference_model.kneighbors(scaled)
        self.local_radius = (
            distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        )
        self.query_model = NearestNeighbors(
            n_neighbors=self.neighbors, metric="euclidean"
        ).fit(scaled)
        self.scale = float(np.median(self.local_radius) + 1e-9)

    def score(self, query: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(query, dtype=float))
        distances, indices = self.query_model.kneighbors(scaled)
        base = distances.mean(axis=1)
        local = self.local_radius[indices].mean(axis=1) + 1e-9
        return np.asarray((base - local) / self.scale, dtype=float)


def _reference_indices(indices: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= _MAX_TRAIN_REFERENCE:
        return values
    return np.sort(
        np.random.default_rng(seed).choice(
            values, size=_MAX_TRAIN_REFERENCE, replace=False
        )
    )


def _score_chunks(
    representation: _Representation,
    model: _LocalGapModel,
    rows: np.ndarray,
    *,
    feature_dimension: int,
) -> np.ndarray:
    parts = [
        model.score(
            _features(
                representation.transform(rows[start : start + _QUERY_CHUNK_SIZE]),
                expected_dimension=feature_dimension,
            )
        )
        for start in range(0, len(rows), _QUERY_CHUNK_SIZE)
    ]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def score_method(
    train_values: np.ndarray,
    test_values: np.ndarray,
    method_name: str,
    seed: int,
) -> ScoreResult:
    train = _rows(train_values, "TRAIN")
    test = _rows(test_values, "TEST")
    if len(train) < _CROSSFIT_FOLDS:
        raise ValueError("at least five TRAIN rows are required")
    if train.shape[1] != test.shape[1]:
        raise ValueError("TRAIN and TEST row shape must match")
    if type(method_name) is not str or method_name not in METHOD_NAMES:
        raise ValueError("method_name must be one of the three approved score sources")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")

    requested_kernels = _REQUESTED_KERNELS[method_name]
    train_oof = np.empty(len(train), dtype=float)
    dimensions: set[int] = set()
    for fold_index, (fit_indices, query_indices) in enumerate(
        crossfit_plan(len(train), seed=seed)
    ):
        reference_indices = _reference_indices(
            fit_indices,
            seed=seed + 1009 * (fold_index + 1) + requested_kernels,
        )
        representation = _make_representation(method_name, seed)
        reference_features = _features(
            representation.fit_transform(train[reference_indices])
        )
        dimensions.add(int(reference_features.shape[1]))
        train_oof[query_indices] = _score_chunks(
            representation,
            _LocalGapModel(reference_features),
            train[query_indices],
            feature_dimension=reference_features.shape[1],
        )

    full_reference_indices = _reference_indices(
        np.arange(len(train), dtype=int),
        seed=seed + 7919 + requested_kernels,
    )
    representation = _make_representation(method_name, seed)
    full_features = _features(
        representation.fit_transform(train[full_reference_indices])
    )
    dimensions.add(int(full_features.shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    test_scores = _score_chunks(
        representation,
        _LocalGapModel(full_features),
        test,
        feature_dimension=full_features.shape[1],
    )
    if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
        raise ValueError("non-finite local-gap scores")
    return ScoreResult(train_oof, test_scores, dimensions.pop(), seed)
