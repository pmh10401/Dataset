"""TRAIN-only cross-fit scoring using official CPU RDST and qualified LocalGap."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .backend_gate_antigravity import BackendNameAntigravity, build_local_gap_backend_antigravity
from .config_antigravity import FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY
from .contracts_antigravity import CandidateScoresAntigravity
from .rdst_antigravity import make_rdst_antigravity


_FOLDS_ANTIGRAVITY = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.crossfit_folds
_CAP_ANTIGRAVITY = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.max_train_reference
_CHUNK_ANTIGRAVITY = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.query_chunk_size
_SEEDS_ANTIGRAVITY = FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.seeds


def _exact_rows_antigravity(value: object, field: str, *, length: int | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.ndim != 2 or value.shape[0] == 0 or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{field} must be a non-empty finite exact 2D numeric ndarray")
    if length is not None and value.shape[1] != length:
        raise ValueError("TRAIN and TEST lengths must match")
    return np.asarray(value, dtype=np.float32)


def crossfit_plan_antigravity(n_train: int, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if type(n_train) is not int or n_train < _FOLDS_ANTIGRAVITY or type(seed) is not int or seed < 0:
        raise ValueError("n_train and seed must be exact valid integers")
    shuffled = np.arange(n_train, dtype=int)
    np.random.default_rng(seed).shuffle(shuffled)
    queries = tuple(np.sort(part) for part in np.array_split(shuffled, _FOLDS_ANTIGRAVITY))
    return tuple((np.sort(np.concatenate(tuple(part for number, part in enumerate(queries) if number != fold))), query) for fold, query in enumerate(queries))


def reference_indices_antigravity(indices: object, seed: int) -> np.ndarray:
    if type(indices) is not np.ndarray or indices.ndim != 1 or len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer) or type(seed) is not int or seed < 0:
        raise ValueError("indices and seed must be exact non-empty integer inputs")
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("reference indices must be unique and non-negative")
    values = np.sort(indices.astype(int, copy=False))
    if len(values) <= _CAP_ANTIGRAVITY:
        return values.copy()
    return np.sort(np.random.default_rng(seed).choice(values, size=_CAP_ANTIGRAVITY, replace=False))


def _features_antigravity(value: object, expected_dimension: int | None = None) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all() or (expected_dimension is not None and values.shape[1] != expected_dimension):
        raise ValueError("RDST features are invalid")
    return values


def _score_rows_antigravity(representation: object, backend: object, rows: np.ndarray, feature_dimension: int) -> np.ndarray:
    parts = []
    for start in range(0, len(rows), _CHUNK_ANTIGRAVITY):
        features = _features_antigravity(representation.transform(rows[start : start + _CHUNK_ANTIGRAVITY, None, :]), feature_dimension)
        parts.append(backend.score(features))
    result = np.concatenate(parts)
    if result.shape != (len(rows),) or not np.isfinite(result).all():
        raise ValueError("LocalGap scores are invalid")
    return result


def score_candidate_antigravity(train_rows: object, test_rows: object, seed: int, backend_name: BackendNameAntigravity) -> CandidateScoresAntigravity:
    train = _exact_rows_antigravity(train_rows, "TRAIN")
    test = _exact_rows_antigravity(test_rows, "TEST", length=train.shape[1])
    if type(seed) is not int or seed not in _SEEDS_ANTIGRAVITY or backend_name not in {"cpu", "mlx", "mps"}:
        raise ValueError("seed or backend name is not frozen")
    oof = np.empty(len(train), dtype=float)
    dimensions: set[int] = set()
    shapelets: set[int] = set()
    for fold, (fit_indices, query_indices) in enumerate(crossfit_plan_antigravity(len(train), seed)):
        reference_indices = reference_indices_antigravity(fit_indices, seed + 1009 * (fold + 1) + 1024)
        representation = make_rdst_antigravity(seed)
        reference = _features_antigravity(representation.fit_transform(train[reference_indices, None, :], y=None))
        dimensions.add(reference.shape[1]); shapelets.add(representation.retained_shapelets)
        backend = build_local_gap_backend_antigravity(backend_name).fit(reference)
        oof[query_indices] = _score_rows_antigravity(representation, backend, train[query_indices], reference.shape[1])
    full_indices = reference_indices_antigravity(np.arange(len(train), dtype=int), seed + 7919 + 1024)
    representation = make_rdst_antigravity(seed)
    reference = _features_antigravity(representation.fit_transform(train[full_indices, None, :], y=None))
    dimensions.add(reference.shape[1]); shapelets.add(representation.retained_shapelets)
    backend = build_local_gap_backend_antigravity(backend_name).fit(reference)
    test_scores = _score_rows_antigravity(representation, backend, test, reference.shape[1])
    if len(dimensions) != 1 or len(shapelets) != 1 or next(iter(dimensions)) != 3 * next(iter(shapelets)) or not np.isfinite(oof).all():
        raise ValueError("fresh RDST metadata is inconsistent")
    return CandidateScoresAntigravity(oof, test_scores, len(train), len(test), next(iter(dimensions)), next(iter(shapelets)), seed)


@dataclass(frozen=True, slots=True)
class AggregatedScoresAntigravity:
    train_oof: np.ndarray
    test_scores: np.ndarray
    expected_train_length: int
    expected_test_length: int
    feature_dimension: int
    retained_shapelets: int
    seeds: tuple[int, int]

    def __post_init__(self) -> None:
        if self.seeds != _SEEDS_ANTIGRAVITY or type(self.train_oof) is not np.ndarray or type(self.test_scores) is not np.ndarray or self.train_oof.ndim != 1 or self.test_scores.ndim != 1 or not np.isfinite(self.train_oof).all() or not np.isfinite(self.test_scores).all():
            raise ValueError("aggregated scores are invalid")
        if self.train_oof.shape != (self.expected_train_length,) or self.test_scores.shape != (self.expected_test_length,) or self.feature_dimension != 3 * self.retained_shapelets:
            raise ValueError("aggregated score metadata differs")
        object.__setattr__(self, "train_oof", np.frombuffer(self.train_oof.astype(float, copy=False).tobytes(), dtype=float))
        object.__setattr__(self, "test_scores", np.frombuffer(self.test_scores.astype(float, copy=False).tobytes(), dtype=float))


def aggregate_seed_scores_antigravity(results: object) -> AggregatedScoresAntigravity:
    if type(results) is not tuple or len(results) != 2 or any(type(item) is not CandidateScoresAntigravity for item in results):
        raise ValueError("results must be exactly two candidate score snapshots")
    first, second = results
    if (first.seed, second.seed) != _SEEDS_ANTIGRAVITY:
        raise ValueError("results must be in the frozen seed order")
    metadata = (first.expected_train_length, first.expected_test_length, first.feature_dimension, first.retained_shapelets)
    if metadata != (second.expected_train_length, second.expected_test_length, second.feature_dimension, second.retained_shapelets):
        raise ValueError("seed results have incompatible metadata")
    return AggregatedScoresAntigravity(np.median(np.vstack((first.train_oof, second.train_oof)), axis=0), np.median(np.vstack((first.test_scores, second.test_scores)), axis=0), *metadata, _SEEDS_ANTIGRAVITY)


__all__ = ["AggregatedScoresAntigravity", "aggregate_seed_scores_antigravity", "crossfit_plan_antigravity", "reference_indices_antigravity", "score_candidate_antigravity"]
