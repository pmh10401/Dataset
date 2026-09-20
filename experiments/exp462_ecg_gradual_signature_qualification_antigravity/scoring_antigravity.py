"""Exp462 Task 3 gradual fold score boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .gradual_features_antigravity import gradual_signature_antigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_PRIMARY_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_order
_MAX_TRAIN_REFERENCE_ANTIGRAVITY = 1000
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


def _feature_matrix_antigravity(rows: np.ndarray, feature_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    matrix = np.vstack([feature_function(row) for row in rows]).astype(np.float64, copy=False)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("gradual feature matrix must be finite and two-dimensional")
    return matrix


def _reference_indexer_antigravity(fit_indices: np.ndarray, *, seed: int, fold_index: int) -> np.ndarray:
    values = np.asarray(fit_indices, dtype=np.int64)
    if len(values) <= _MAX_TRAIN_REFERENCE_ANTIGRAVITY:
        return values
    selected = np.random.default_rng(seed + 1009 * (fold_index + 1) + 24).choice(
        values,
        size=_MAX_TRAIN_REFERENCE_ANTIGRAVITY,
        replace=False,
    )
    return np.sort(selected)


def _coerce_reference_indices_antigravity(values: object, train_rows: int) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.size < 2:
        raise ValueError("reference indices must be a one-dimensional ndarray with at least two rows")
    if not np.issubdtype(values.dtype, np.integer) or values.dtype == np.dtype(bool):
        raise ValueError("reference indices must use an integer dtype")
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if np.any(result < 0) or np.any(result >= train_rows):
        raise ValueError("reference indices must be in bounds")
    if len(np.unique(result)) != len(result):
        raise ValueError("reference indices must be unique")
    result.setflags(write=False)
    return result


def _gradual_fold_scorer_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    local_gap_model: Callable[..., object],
    feature_function: Callable[[np.ndarray], np.ndarray],
    reference_indexer: Callable[..., object],
) -> tuple[np.ndarray, Mapping[str, np.ndarray], int]:
    reference_indices = _coerce_reference_indices_antigravity(
        reference_indexer(fit_indices, seed=seed, fold_index=fold_index),
        len(train),
    )
    reference_features = _feature_matrix_antigravity(train[reference_indices], feature_function)
    model = local_gap_model(reference_features, neighbors=FROZEN_CONFIG_ANTIGRAVITY.knn_k)
    query_features = _feature_matrix_antigravity(train[query_indices], feature_function)
    original = np.asarray(model.score(query_features), dtype=np.float64)
    primary_rows = tuple(_build_primary_rows_antigravity(row) for row in train[query_indices])
    primary: dict[str, np.ndarray] = {}
    for change_type in _PRIMARY_TYPES_ANTIGRAVITY:
        rows = np.stack([row[change_type] for row in primary_rows])
        primary[change_type] = np.asarray(
            model.score(_feature_matrix_antigravity(rows, feature_function)),
            dtype=np.float64,
        )
    return original, MappingProxyType(primary), int(reference_features.shape[1])


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


def _coerce_baseline_antigravity(value: object, query_len: int) -> tuple[np.ndarray, Mapping[str, np.ndarray]]:
    if not all(hasattr(value, attr) for attr in ("methods", "original", "primary")):
        raise ValueError("Exp461 scorer returned an invalid result")
    methods = getattr(value, "methods")
    original = getattr(value, "original")
    primary = getattr(value, "primary")
    if type(methods) is not tuple or "hydra_period_rank_max" not in methods:
        raise ValueError("Exp461 scorer did not return the reviewed branch")
    if not isinstance(original, Mapping) or not isinstance(primary, Mapping):
        raise ValueError("Exp461 score containers are invalid")
    baseline_original = _readonly_vector_antigravity(
        original["hydra_period_rank_max"],
        "Exp461 hydra_period_rank_max original",
    )
    if len(baseline_original) != query_len:
        raise ValueError("Exp461 original length must match query size")
    baseline_primary = _copy_primary_antigravity(
        primary["hydra_period_rank_max"],
        query_len,
        "Exp461 hydra_period_rank_max primary",
    )
    return baseline_original, baseline_primary


def _score_fold_methods_with_dependencies_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    baseline_fold_scorer: Callable[..., object],
    local_gap_model: Callable[..., object],
    feature_function: Callable[[np.ndarray], np.ndarray],
    reference_indexer: Callable[..., object],
) -> FoldMethodScoresAntigravity:
    train_matrix = _train_antigravity(train)
    fit = _indices_antigravity(fit_indices, len(train_matrix), "fit_indices")
    query = _indices_antigravity(query_indices, len(train_matrix), "query_indices")
    if np.intersect1d(fit, query).size:
        raise ValueError("fit and query indices must be disjoint")
    seed_value = _seed_antigravity(seed)
    fold_value = _fold_index_antigravity(fold_index)

    baseline_original, baseline_primary = _coerce_baseline_antigravity(
        baseline_fold_scorer(
            train_matrix,
            fit,
            query,
            seed=seed_value,
            fold_index=fold_value,
        ),
        len(query),
    )
    gradual_original, gradual_primary, gradual_dimension = _gradual_fold_scorer_antigravity(
        train_matrix,
        fit,
        query,
        seed=seed_value,
        fold_index=fold_value,
        local_gap_model=local_gap_model,
        feature_function=feature_function,
        reference_indexer=reference_indexer,
    )

    # Exp472: use raw distance instead of rank to avoid [0,1] saturation.
    hybrid_original = np.maximum(baseline_original, gradual_original)
    hybrid_primary = {
        change_type: np.maximum(
            baseline_primary[change_type],
            gradual_primary[change_type],
        )
        for change_type in _PRIMARY_TYPES_ANTIGRAVITY
    }
    return FoldMethodScoresAntigravity(
        methods=_METHODS_ANTIGRAVITY,
        original={
            "hydra_period_rank_max": baseline_original,
            "gradual_signature_local_gap": gradual_original,
            "hydra_period_gradual_rank_max": hybrid_original,
        },
        primary={
            "hydra_period_rank_max": baseline_primary,
            "gradual_signature_local_gap": gradual_primary,
            "hydra_period_gradual_rank_max": hybrid_primary,
        },
        feature_dimensions={
            "hydra_period_rank_max": 2,
            "gradual_signature_local_gap": gradual_dimension,
            "hydra_period_gradual_rank_max": 2,
        },
        provenance={
            "hydra_period_rank_max": "exp461_reviewed_delegate",
            "gradual_signature_local_gap": "exp462_gradual_signature_local_gap",
            "hydra_period_gradual_rank_max": "exp462_rank_max",
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
    from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
        LocalGapModelAntigravity,
    )
    from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
        score_fold_methods_antigravity as score_exp461_fold_methods_antigravity,
    )

    return _score_fold_methods_with_dependencies_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
        baseline_fold_scorer=score_exp461_fold_methods_antigravity,
        local_gap_model=LocalGapModelAntigravity,
        feature_function=gradual_signature_antigravity,
        reference_indexer=_reference_indexer_antigravity,
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "anomaly_rank_antigravity",
    "score_fold_methods_antigravity",
)
