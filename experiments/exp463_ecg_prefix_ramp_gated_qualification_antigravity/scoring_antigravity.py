"""Exp463 Task 3 fold scores for the prefix-ramp comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .ramp_features_antigravity import (
    prefix_ramp_score_antigravity,
    template_residual_ramp_score_antigravity,
    whole_series_ramp_score_antigravity,
)


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_order
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
    expected_shape = (
        FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows,
        FROZEN_CONFIG_ANTIGRAVITY.normal2_width,
    )
    if train.shape[1] != expected_shape[1]:
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


def _copy_primary_antigravity(
    values: Mapping[str, object],
    expected_len: int,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or tuple(values) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve change type order")
    frozen: dict[str, np.ndarray] = {}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        vector = _readonly_vector_antigravity(values[change_type], f"{name}[{change_type}]")
        if len(vector) != expected_len:
            raise ValueError(f"{name} score lengths must match")
        frozen[change_type] = vector
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    """Immutable M0–M6 score evidence for one seed/fold query."""

    methods: tuple[str, ...]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    feature_dimensions: Mapping[str, int]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.methods) is not tuple or self.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("methods must preserve frozen M0-M6 order")
        if not isinstance(self.original, Mapping) or tuple(self.original) != self.methods:
            raise ValueError("original must preserve method order")
        if not isinstance(self.primary, Mapping) or tuple(self.primary) != self.methods:
            raise ValueError("primary must preserve method order")
        if not isinstance(self.feature_dimensions, Mapping) or tuple(self.feature_dimensions) != self.methods:
            raise ValueError("feature_dimensions must preserve method order")
        if not isinstance(self.provenance, Mapping) or tuple(self.provenance) != self.methods:
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
    """Return (1 + count(reference <= score)) / (n + 1) for each query."""

    reference = _readonly_vector_antigravity(reference_scores, "reference_scores")
    query = _readonly_vector_antigravity(scores, "scores")
    counts = np.count_nonzero(reference[:, None] <= query[None, :], axis=0)
    result = np.array(
        (1.0 + counts) / (len(reference) + 1.0),
        dtype=np.float64,
        copy=True,
        order="C",
    )
    if not np.isfinite(result).all():
        raise ValueError("rank output must be finite")
    result.setflags(write=False)
    return result


def _same_length_vectors_antigravity(
    left: np.ndarray,
    right: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_copy = _readonly_vector_antigravity(left, f"{name}.left")
    right_copy = _readonly_vector_antigravity(right, f"{name}.right")
    if len(left_copy) != len(right_copy):
        raise ValueError(f"{name} vectors must have equal lengths")
    return left_copy, right_copy


def _gated_max_antigravity(baseline: np.ndarray, ramp_rank: np.ndarray) -> np.ndarray:
    """Apply the fixed 0.995 gate, retaining baseline below the threshold."""

    base, ramp = _same_length_vectors_antigravity(baseline, ramp_rank, "gated_max")
    result = np.where(
        ramp >= FROZEN_CONFIG_ANTIGRAVITY.gate,
        np.maximum(base, ramp),
        base,
    ).astype(np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _always_max_antigravity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_copy, right_copy = _same_length_vectors_antigravity(left, right, "always_max")
    result = np.maximum(left_copy, right_copy).astype(np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _primary_rows_antigravity(values: np.ndarray) -> Mapping[str, np.ndarray]:
    row = np.asarray(values, dtype=np.float64)
    center = float(np.median(row))
    robust_scale = float(_MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(row - center)))
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("row robust scale must be positive and finite")
    onset = FROZEN_CONFIG_ANTIGRAVITY.onset
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
                _CHANGE_TYPES_ANTIGRAVITY,
                (level, drift, amplitude, warped),
                strict=True,
            )
        )
    )


def _feature_scores_antigravity(
    rows: np.ndarray,
    feature_function: Callable[[np.ndarray], float],
) -> np.ndarray:
    result = np.asarray([feature_function(row) for row in rows], dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or not np.isfinite(result).all():
        raise ValueError("feature scores must be finite and non-empty")
    return result


def _ramp_fold_evidence_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    feature_function: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    fit_raw = _feature_scores_antigravity(train[fit_indices], feature_function)
    query_raw = _feature_scores_antigravity(train[query_indices], feature_function)
    query_rows = tuple(_primary_rows_antigravity(row) for row in train[query_indices])
    primary_raw: dict[str, np.ndarray] = {}
    primary_rank: dict[str, np.ndarray] = {}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        rows = np.stack([row[change_type] for row in query_rows])
        raw = _feature_scores_antigravity(rows, feature_function)
        primary_raw[change_type] = raw
        primary_rank[change_type] = anomaly_rank_antigravity(fit_raw, raw)
    query_rank = anomaly_rank_antigravity(fit_raw, query_raw)
    return (
        query_raw,
        query_rank,
        MappingProxyType(primary_raw),
        MappingProxyType(primary_rank),
    )


def _template_ramp_fold_evidence_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    template = np.median(np.asarray(train[fit_indices], dtype=np.float64), axis=0)
    if template.shape != (FROZEN_CONFIG_ANTIGRAVITY.normal2_width,) or not np.isfinite(template).all():
        raise ValueError("TRAIN-fit median template must be finite with width 140")
    feature = lambda row: template_residual_ramp_score_antigravity(row, template)
    return _ramp_fold_evidence_antigravity(
        train,
        fit_indices,
        query_indices,
        feature_function=feature,
    )


def _coerce_branch_antigravity(
    value: object,
    query_len: int,
    expected_method: str,
    name: str,
) -> tuple[np.ndarray, Mapping[str, np.ndarray], int, str]:
    if not all(hasattr(value, attr) for attr in ("methods", "original", "primary")):
        raise ValueError(f"{name} scorer returned an invalid result")
    methods = getattr(value, "methods")
    original = getattr(value, "original")
    primary = getattr(value, "primary")
    if type(methods) is not tuple or expected_method not in methods:
        raise ValueError(f"{name} scorer did not return the reviewed branch")
    if not isinstance(original, Mapping) or not isinstance(primary, Mapping):
        raise ValueError(f"{name} score containers are invalid")
    original_vector = _readonly_vector_antigravity(
        original[expected_method],
        f"{name} original",
    )
    if len(original_vector) != query_len:
        raise ValueError(f"{name} original length must match query size")
    primary_vectors = _copy_primary_antigravity(
        primary[expected_method],
        query_len,
        f"{name} primary",
    )
    dimensions = getattr(value, "feature_dimensions", {})
    dimension = dimensions.get(expected_method, 1) if isinstance(dimensions, Mapping) else 1
    if type(dimension) is not int or dimension < 1:
        raise ValueError(f"{name} feature dimension must be positive")
    provenance = getattr(value, "provenance", {})
    source = provenance.get(expected_method, name) if isinstance(provenance, Mapping) else name
    if type(source) is not str or not source:
        raise ValueError(f"{name} provenance must be non-empty")
    return original_vector, primary_vectors, dimension, source


def _score_fold_methods_with_dependencies_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    baseline_fold_scorer: Callable[..., object],
    replay_fold_scorer: Callable[..., object],
) -> FoldMethodScoresAntigravity:
    train_matrix = _train_antigravity(train)
    fit = _indices_antigravity(fit_indices, len(train_matrix), "fit_indices")
    query = _indices_antigravity(query_indices, len(train_matrix), "query_indices")
    if np.intersect1d(fit, query).size:
        raise ValueError("fit and query indices must be disjoint")
    seed_value = _seed_antigravity(seed)
    fold_value = _fold_index_antigravity(fold_index)

    baseline_original, baseline_primary, baseline_dimension, baseline_source = _coerce_branch_antigravity(
        baseline_fold_scorer(
            train_matrix,
            fit,
            query,
            seed=seed_value,
            fold_index=fold_value,
        ),
        len(query),
        "hydra_period_rank_max",
        "M0",
    )
    replay_original, replay_primary, replay_dimension, replay_source = _coerce_branch_antigravity(
        replay_fold_scorer(
            train_matrix,
            fit,
            query,
            seed=seed_value,
            fold_index=fold_value,
        ),
        len(query),
        "hydra_period_gradual_rank_max",
        "M6",
    )

    m1_original, m1_rank, m1_primary_raw, m1_primary_rank = _ramp_fold_evidence_antigravity(
        train_matrix,
        fit,
        query,
        feature_function=prefix_ramp_score_antigravity,
    )
    _, m4_rank, _, m4_primary_rank = _template_ramp_fold_evidence_antigravity(
        train_matrix,
        fit,
        query,
    )
    m5_original, m5_rank, m5_primary_raw, m5_primary_rank = _ramp_fold_evidence_antigravity(
        train_matrix,
        fit,
        query,
        feature_function=whole_series_ramp_score_antigravity,
    )

    m2_original = _gated_max_antigravity(baseline_original, m1_rank)
    m3_original = _always_max_antigravity(baseline_original, m1_rank)
    m4_original = _gated_max_antigravity(baseline_original, m4_rank)
    m5_original = _gated_max_antigravity(baseline_original, m5_rank)
    # The reviewed Exp462 branch already contains its own LocalGap rank-max
    # semantics. Preserve that replay control exactly; do not combine it with
    # the Exp461 M0 a second time here.
    m6_original = replay_original

    m2_primary = MappingProxyType(
        {
            change_type: _gated_max_antigravity(
                baseline_primary[change_type],
                m1_primary_rank[change_type],
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )
    m3_primary = MappingProxyType(
        {
            change_type: _always_max_antigravity(
                baseline_primary[change_type],
                m1_primary_rank[change_type],
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )
    m4_primary = MappingProxyType(
        {
            change_type: _gated_max_antigravity(
                baseline_primary[change_type],
                m4_primary_rank[change_type],
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )
    m5_primary = MappingProxyType(
        {
            change_type: _gated_max_antigravity(
                baseline_primary[change_type],
                m5_primary_rank[change_type],
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )
    m6_primary = replay_primary

    return FoldMethodScoresAntigravity(
        methods=_METHODS_ANTIGRAVITY,
        original={
            "M0": baseline_original,
            "M1": m1_original,
            "M2": m2_original,
            "M3": m3_original,
            "M4": m4_original,
            "M5": m5_original,
            "M6": m6_original,
        },
        primary={
            "M0": baseline_primary,
            "M1": m1_primary_raw,
            "M2": m2_primary,
            "M3": m3_primary,
            "M4": m4_primary,
            "M5": m5_primary,
            "M6": m6_primary,
        },
        feature_dimensions={
            "M0": baseline_dimension,
            "M1": 1,
            "M2": 2,
            "M3": 2,
            "M4": 1,
            "M5": 1,
            "M6": replay_dimension,
        },
        provenance={
            "M0": baseline_source,
            "M1": "exp463_prefix_ramp",
            "M2": "exp463_prefix_ramp_gated_max",
            "M3": "exp463_prefix_ramp_always_max",
            "M4": "exp463_template_residual_ramp_gated_max",
            "M5": "exp463_whole_series_ramp_gated_max",
            "M6": replay_source,
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
    from experiments.exp461_ecg_period_signature_qualification_antigravity.scoring_antigravity import (
        score_fold_methods_antigravity as score_exp461_fold_methods_antigravity,
    )
    from experiments.exp462_ecg_gradual_signature_qualification_antigravity.scoring_antigravity import (
        score_fold_methods_antigravity as score_exp462_fold_methods_antigravity,
    )

    return _score_fold_methods_with_dependencies_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
        baseline_fold_scorer=score_exp461_fold_methods_antigravity,
        replay_fold_scorer=score_exp462_fold_methods_antigravity,
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "anomaly_rank_antigravity",
    "score_fold_methods_antigravity",
)
