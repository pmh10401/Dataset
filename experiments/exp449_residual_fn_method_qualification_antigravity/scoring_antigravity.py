"""Five-fold label-free LocalGap scores for the frozen Exp449 sources."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts_antigravity import MethodCandidateAntigravity
from .local_gap_antigravity import LocalGapModelAntigravity
from .representations_antigravity import make_representation_antigravity


KNN_NEIGHBORS_ANTIGRAVITY = 3
MAX_TRAIN_REFERENCE_ANTIGRAVITY = 1000
QUERY_CHUNK_SIZE_ANTIGRAVITY = 128
CROSSFIT_FOLDS_ANTIGRAVITY = 5
_REQUESTED_KERNELS_ANTIGRAVITY = {
    "minirocket_local_gap": 4096,
    "multirocket_local_gap": 1024,
    "hydra_local_gap": 8,
    "hydra_multirocket_local_gap": 1024,
}


@dataclass(frozen=True, slots=True)
class CandidateScoresAntigravity:
    train_oof: np.ndarray
    test_scores: np.ndarray
    feature_dimension: int
    seed: int

    def __post_init__(self) -> None:
        if type(self.feature_dimension) is not int or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be an exact positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be an exact non-negative integer")
        for field in ("train_oof", "test_scores"):
            values = getattr(self, field)
            if type(values) is not np.ndarray or values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError(f"{field} must be a finite one-dimensional ndarray")
            object.__setattr__(self, field, np.frombuffer(values.astype(float, copy=False).tobytes(), dtype=float))


def crossfit_plan_antigravity(n_train: int, *, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if type(n_train) is not int or n_train < CROSSFIT_FOLDS_ANTIGRAVITY:
        raise ValueError("at least five TRAIN rows are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    shuffled = np.arange(n_train, dtype=int)
    np.random.default_rng(seed).shuffle(shuffled)
    queries = tuple(np.sort(part) for part in np.array_split(shuffled, CROSSFIT_FOLDS_ANTIGRAVITY))
    return tuple((np.sort(np.concatenate([part for other, part in enumerate(queries) if other != index])), query) for index, query in enumerate(queries))


def _reference_indices_antigravity(indices: np.ndarray, *, seed: int) -> np.ndarray:
    values = np.asarray(indices, dtype=int)
    if len(values) <= MAX_TRAIN_REFERENCE_ANTIGRAVITY:
        return values
    selected = np.random.default_rng(seed).choice(values, size=MAX_TRAIN_REFERENCE_ANTIGRAVITY, replace=False)
    return np.sort(selected)


def _feature_matrix_antigravity(values: object, *, expected_dimension: int | None = None) -> np.ndarray:
    features = np.asarray(values, dtype=float)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("non-finite representation features")
    if expected_dimension is not None and features.shape[1] != expected_dimension:
        raise ValueError("inconsistent feature dimensions")
    return features


def _score_chunks_antigravity(representation: object, model: LocalGapModelAntigravity, rows: np.ndarray, *, feature_dimension: int) -> np.ndarray:
    parts = [
        model.score(_feature_matrix_antigravity(representation.transform(rows[start : start + QUERY_CHUNK_SIZE_ANTIGRAVITY]), expected_dimension=feature_dimension))
        for start in range(0, len(rows), QUERY_CHUNK_SIZE_ANTIGRAVITY)
    ]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _rows_antigravity(rows: np.ndarray, field: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{field} must be a non-empty finite two-dimensional array")
    return values


def score_candidate_antigravity(train_rows: np.ndarray, test_rows: np.ndarray, candidate: MethodCandidateAntigravity, *, seed: int) -> CandidateScoresAntigravity:
    """Score without labels, refitting the frozen representation on each LocalGap reference."""
    train = _rows_antigravity(train_rows, "TRAIN")
    test = _rows_antigravity(test_rows, "TEST")
    if train.shape[1] != test.shape[1]:
        raise ValueError("TRAIN and TEST row shape must match")
    if type(candidate) is not MethodCandidateAntigravity:
        raise ValueError("candidate must be an exact MethodCandidateAntigravity")
    if candidate.name != candidate.representation:
        raise ValueError("candidate name and representation must match")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    if seed not in candidate.seeds:
        raise ValueError("seed must be declared in candidate.seeds")
    requested_kernels = _REQUESTED_KERNELS_ANTIGRAVITY[candidate.representation]
    train_oof = np.empty(len(train), dtype=float)
    dimensions: set[int] = set()
    for fold_index, (fit_indices, query_indices) in enumerate(crossfit_plan_antigravity(len(train), seed=seed)):
        reference_indices = _reference_indices_antigravity(
            fit_indices, seed=seed + 1009 * (fold_index + 1) + requested_kernels
        )
        representation = make_representation_antigravity(candidate.representation, seed)
        reference_features = _feature_matrix_antigravity(representation.fit_transform(train[reference_indices]))
        dimensions.add(int(reference_features.shape[1]))
        train_oof[query_indices] = _score_chunks_antigravity(
            representation, LocalGapModelAntigravity(reference_features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY), train[query_indices], feature_dimension=reference_features.shape[1]
        )
    full_reference_indices = _reference_indices_antigravity(
        np.arange(len(train), dtype=int), seed=seed + 7919 + requested_kernels
    )
    representation = make_representation_antigravity(candidate.representation, seed)
    full_features = _feature_matrix_antigravity(representation.fit_transform(train[full_reference_indices]))
    dimensions.add(int(full_features.shape[1]))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    test_scores = _score_chunks_antigravity(
        representation, LocalGapModelAntigravity(full_features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY), test, feature_dimension=full_features.shape[1]
    )
    if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
        raise ValueError("non-finite local-gap scores")
    return CandidateScoresAntigravity(train_oof, test_scores, dimensions.pop(), seed)
