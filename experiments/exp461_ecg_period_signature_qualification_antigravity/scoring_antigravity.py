"""Exp461 Task 3 fold score boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
    LocalGapModelAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    _reference_indices_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .period_features_antigravity import period_signature_antigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_PRIMARY_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.primary_types
_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _readonly_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _train_antigravity(train: object) -> np.ndarray:
    if type(train) is not np.ndarray or train.ndim != 2:
        raise ValueError("train must be a two-dimensional ndarray")
    if train.shape[1] != FROZEN_CONFIG_ANTIGRAVITY.normal2_width:
        raise ValueError("train must have width 140")
    if train.dtype != np.dtype(np.float32):
        raise ValueError("train must be float32")
    if not np.isfinite(train).all():
        raise ValueError("train must be finite")
    return train


def _indices_antigravity(values: object, train_rows: int, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if not np.issubdtype(values.dtype, np.integer) or values.dtype == np.dtype(bool):
        raise ValueError(f"{name} must use an integer dtype")
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if np.any(result < 0) or np.any(result >= train_rows):
        raise ValueError(f"{name} must be in bounds")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    if not np.array_equal(result, np.sort(result)):
        raise ValueError(f"{name} must be sorted")
    result.setflags(write=False)
    return result


def _seed_antigravity(seed: object) -> int:
    if type(seed) is not int or seed not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the frozen seeds")
    return seed


def _fold_index_antigravity(fold_index: object) -> int:
    if type(fold_index) is not int or not 0 <= fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be one of the frozen folds")
    return fold_index


def _copy_primary_antigravity(values: Mapping[str, object], expected_len: int, name: str) -> Mapping[str, np.ndarray]:
    if tuple(values) != _PRIMARY_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve primary type order")
    frozen: dict[str, np.ndarray] = {}
    for change_type in _PRIMARY_TYPES_ANTIGRAVITY:
        vector = _readonly_vector_antigravity(values[change_type], f"{name}[{change_type}]")
        if len(vector) != expected_len:
            raise ValueError(f"{name} score lengths must match")
        frozen[change_type] = vector
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    methods: tuple[str, ...]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    feature_dimensions: Mapping[str, int]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.methods) is not tuple or self.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("methods must preserve the frozen order")
        if tuple(self.original) != self.methods:
            raise ValueError("original must preserve method order")
        if tuple(self.primary) != self.methods:
            raise ValueError("primary must preserve method order")
        if tuple(self.feature_dimensions) != self.methods:
            raise ValueError("feature_dimensions must preserve method order")
        if tuple(self.provenance) != self.methods:
            raise ValueError("provenance must preserve method order")

        frozen_original: dict[str, np.ndarray] = {}
        frozen_primary: dict[str, Mapping[str, np.ndarray]] = {}
        expected_len: int | None = None
        for method in self.methods:
            original = _readonly_vector_antigravity(self.original[method], f"original[{method}]")
            if expected_len is None:
                expected_len = len(original)
            elif len(original) != expected_len:
                raise ValueError("method original lengths must match")
            frozen_original[method] = original
            frozen_primary[method] = _copy_primary_antigravity(
                self.primary[method],
                len(original),
                f"primary[{method}]",
            )

        frozen_dimensions: dict[str, int] = {}
        frozen_provenance: dict[str, str] = {}
        for method in self.methods:
            dimension = self.feature_dimensions[method]
            if type(dimension) is not int or dimension < 1:
                raise ValueError("feature dimensions must be positive integers")
            provenance = self.provenance[method]
            if type(provenance) is not str or not provenance:
                raise ValueError("provenance values must be non-empty strings")
            frozen_dimensions[method] = dimension
            frozen_provenance[method] = provenance

        object.__setattr__(self, "original", MappingProxyType(frozen_original))
        object.__setattr__(self, "primary", MappingProxyType(frozen_primary))
        object.__setattr__(self, "feature_dimensions", MappingProxyType(frozen_dimensions))
        object.__setattr__(self, "provenance", MappingProxyType(frozen_provenance))


def anomaly_rank_antigravity(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    reference = _readonly_vector_antigravity(reference_scores, "reference_scores")
    query = _readonly_vector_antigravity(scores, "scores")
    counts = np.count_nonzero(reference[:, None] <= query[None, :], axis=0)
    result = np.array((1.0 + counts) / (len(reference) + 1.0), dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("rank output must be finite")
    result.setflags(write=False)
    return result


def _build_primary_rows_antigravity(values: np.ndarray) -> Mapping[str, np.ndarray]:
    row = np.asarray(values, dtype=np.float64)
    center = float(np.median(row))
    robust_scale = float(_MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(row - center)))
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("row robust scale must be positive and finite")
    onset = int(np.floor(0.40 * row.size))
    duration = row.size - onset

    level = np.array(row, copy=True)
    level[onset:] += 3.0 * robust_scale

    drift = np.array(row, copy=True)
    drift[onset:] += np.linspace(0.0, 3.0 * robust_scale, duration, dtype=np.float64)

    amplitude = center + 1.5 * (row - center)

    warped = np.array(row, copy=True)
    region = row[onset:]
    source = np.arange(duration, dtype=np.float64)
    warped[onset:] = np.interp(source / 1.25, source, region)

    return MappingProxyType(
        dict(
            zip(
                _PRIMARY_TYPES_ANTIGRAVITY,
                (
                    np.asarray(level, dtype=np.float64),
                    np.asarray(drift, dtype=np.float64),
                    np.asarray(amplitude, dtype=np.float64),
                    np.asarray(warped, dtype=np.float64),
                ),
                strict=True,
            )
        )
    )


def _period_feature_matrix_antigravity(rows: np.ndarray) -> np.ndarray:
    matrix = np.vstack([period_signature_antigravity(row) for row in rows]).astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[1] != 120 or not np.isfinite(matrix).all():
        raise ValueError("period feature matrix must be finite with 120 columns")
    return matrix


def _period_fold_scorer_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
) -> tuple[np.ndarray, Mapping[str, np.ndarray], int]:
    reference_indices = _reference_indices_antigravity(
        fit_indices,
        seed=seed + 1009 * (fold_index + 1) + FROZEN_CONFIG_ANTIGRAVITY.kernels,
    )
    reference_features = _period_feature_matrix_antigravity(train[reference_indices])
    model = LocalGapModelAntigravity(
        reference_features,
        neighbors=FROZEN_CONFIG_ANTIGRAVITY.knn_k,
    )
    original = np.asarray(
        model.score(_period_feature_matrix_antigravity(train[query_indices])),
        dtype=np.float64,
    )
    row_primary = tuple(_build_primary_rows_antigravity(row) for row in train[query_indices])
    primary: dict[str, np.ndarray] = {}
    for change_type in _PRIMARY_TYPES_ANTIGRAVITY:
        rows = np.stack([row[change_type] for row in row_primary])
        primary[change_type] = np.asarray(
            model.score(_period_feature_matrix_antigravity(rows)),
            dtype=np.float64,
        )
    return original, MappingProxyType(primary), 120


def _hydra_fold_scorer_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
) -> tuple[np.ndarray, Mapping[str, np.ndarray], int]:
    from experiments.exp459_ecg_normal2_blocked_validation_antigravity.qualification_antigravity import (
        _score_fold_antigravity,
    )

    fold = _score_fold_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
    )
    return np.asarray(fold.original, dtype=np.float64), fold.primary, int(fold.feature_dimension)


def _coerce_branch_antigravity(value: object, query_len: int, name: str) -> tuple[np.ndarray, Mapping[str, np.ndarray], int]:
    if type(value) is tuple and len(value) == 3:
        original, primary, dimension = value
    elif all(hasattr(value, attr) for attr in ("original", "primary", "feature_dimension")):
        original = getattr(value, "original")
        primary = getattr(value, "primary")
        dimension = getattr(value, "feature_dimension")
    else:
        raise ValueError(f"{name} scorer returned an invalid result")
    original_vector = _readonly_vector_antigravity(original, f"{name}.original")
    if len(original_vector) != query_len:
        raise ValueError(f"{name} original length must match query size")
    if not isinstance(primary, Mapping):
        raise ValueError(f"{name} primary must be a mapping")
    primary_vectors = _copy_primary_antigravity(primary, query_len, f"{name}.primary")
    if type(dimension) is not int or dimension < 1:
        raise ValueError(f"{name} feature dimension must be a positive integer")
    return original_vector, primary_vectors, dimension


def _score_fold_methods_with_dependencies_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    hydra_fold_scorer: Callable[..., object],
    period_fold_scorer: Callable[..., object],
) -> FoldMethodScoresAntigravity:
    train_matrix = _train_antigravity(train)
    fit = _indices_antigravity(fit_indices, len(train_matrix), "fit_indices")
    query = _indices_antigravity(query_indices, len(train_matrix), "query_indices")
    if np.intersect1d(fit, query).size:
        raise ValueError("fit and query indices must be disjoint")
    seed_value = _seed_antigravity(seed)
    fold_value = _fold_index_antigravity(fold_index)

    hydra_original, hydra_primary, hydra_dimension = _coerce_branch_antigravity(
        hydra_fold_scorer(
            train_matrix,
            fit,
            query,
            seed=seed_value,
            fold_index=fold_value,
        ),
        len(query),
        "hydra",
    )
    period_original, period_primary, period_dimension = _coerce_branch_antigravity(
        period_fold_scorer(
            train_matrix,
            fit,
            query,
            seed=seed_value,
            fold_index=fold_value,
        ),
        len(query),
        "period",
    )

    hybrid_original = np.maximum(
        anomaly_rank_antigravity(hydra_original, hydra_original),
        anomaly_rank_antigravity(period_original, period_original),
    )
    hybrid_primary = {
        change_type: np.maximum(
            anomaly_rank_antigravity(hydra_original, hydra_primary[change_type]),
            anomaly_rank_antigravity(period_original, period_primary[change_type]),
        )
        for change_type in _PRIMARY_TYPES_ANTIGRAVITY
    }
    return FoldMethodScoresAntigravity(
        methods=_METHODS_ANTIGRAVITY,
        original={
            "hydra_local_gap": hydra_original,
            "period_signature_local_gap": period_original,
            "hydra_period_rank_max": hybrid_original,
        },
        primary={
            "hydra_local_gap": hydra_primary,
            "period_signature_local_gap": period_primary,
            "hydra_period_rank_max": hybrid_primary,
        },
        feature_dimensions={
            "hydra_local_gap": hydra_dimension,
            "period_signature_local_gap": period_dimension,
            "hydra_period_rank_max": 2,
        },
        provenance={
            "hydra_local_gap": "exp459_hydra_local_gap",
            "period_signature_local_gap": "exp461_period_signature_local_gap",
            "hydra_period_rank_max": "exp461_branch_rank_max",
        },
    )


def score_fold_methods_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
) -> FoldMethodScoresAntigravity:
    return _score_fold_methods_with_dependencies_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
        hydra_fold_scorer=_hydra_fold_scorer_antigravity,
        period_fold_scorer=_period_fold_scorer_antigravity,
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "anomaly_rank_antigravity",
    "score_fold_methods_antigravity",
)
