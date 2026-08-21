"""Exp464 Task 3 fold scores for centered residual ramp confirmation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .centered_ramp_antigravity import centered_ramp_evidence_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_CONFIRMATION_TYPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
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


def _readonly_optional_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional ndarray")
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
    if not np.issubdtype(train.dtype, np.floating):
        raise ValueError("train must use a floating dtype")
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


def _copy_confirmation_counts_antigravity(
    values: Mapping[str, object],
    query_len: int,
    name: str,
) -> Mapping[str, int]:
    if not isinstance(values, Mapping) or tuple(values) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve confirmation count order")
    frozen: dict[str, int] = {}
    for key in _CONFIRMATION_TYPES_ANTIGRAVITY:
        value = values[key]
        if type(value) is not int or not 0 <= value <= query_len:
            raise ValueError(f"{name}[{key}] must be an integer in [0, query_len]")
        frozen[key] = value
    return MappingProxyType(frozen)


def _copy_query_evidence_antigravity(
    values: Mapping[str, object],
    expected_len: int,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(values) == ():
        return MappingProxyType({})
    if tuple(values) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve evidence scope order")
    frozen: dict[str, np.ndarray] = {}
    for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
        vector = _readonly_vector_antigravity(values[scope], f"{name}[{scope}]")
        if len(vector) != expected_len:
            raise ValueError(f"{name}[{scope}] length must match query size")
        frozen[scope] = vector
    return MappingProxyType(frozen)


def anomaly_rank_antigravity(
    reference_scores: np.ndarray,
    scores: np.ndarray,
) -> np.ndarray:
    reference = _readonly_vector_antigravity(reference_scores, "reference_scores")
    query = _readonly_vector_antigravity(scores, "scores")
    counts = np.count_nonzero(reference[:, None] <= query[None, :], axis=0)
    result = np.array(
        (1.0 + counts) / (len(reference) + 1.0),
        dtype=np.float64,
        copy=True,
        order="C",
    )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    methods: tuple[str, ...]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    confirmation_counts: Mapping[str, Mapping[str, int]]
    fit_amplitudes: Mapping[str, np.ndarray]
    query_amplitudes: Mapping[str, Mapping[str, np.ndarray]]
    query_cosines: Mapping[str, Mapping[str, np.ndarray]]
    feature_dimensions: Mapping[str, int]
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.methods) is not tuple or self.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("methods must preserve frozen C0-C8 order")
        if not isinstance(self.original, Mapping) or tuple(self.original) != self.methods:
            raise ValueError("original must preserve method order")
        if not isinstance(self.primary, Mapping) or tuple(self.primary) != self.methods:
            raise ValueError("primary must preserve method order")
        if not isinstance(self.confirmation_counts, Mapping) or tuple(self.confirmation_counts) != self.methods:
            raise ValueError("confirmation_counts must preserve method order")
        if not isinstance(self.fit_amplitudes, Mapping) or tuple(self.fit_amplitudes) != self.methods:
            raise ValueError("fit_amplitudes must preserve method order")
        if not isinstance(self.query_amplitudes, Mapping) or tuple(self.query_amplitudes) != self.methods:
            raise ValueError("query_amplitudes must preserve method order")
        if not isinstance(self.query_cosines, Mapping) or tuple(self.query_cosines) != self.methods:
            raise ValueError("query_cosines must preserve method order")
        if not isinstance(self.feature_dimensions, Mapping) or tuple(self.feature_dimensions) != self.methods:
            raise ValueError("feature_dimensions must preserve method order")
        if not isinstance(self.provenance, Mapping) or tuple(self.provenance) != self.methods:
            raise ValueError("provenance must preserve method order")

        frozen_original: dict[str, np.ndarray] = {}
        frozen_primary: dict[str, Mapping[str, np.ndarray]] = {}
        frozen_counts: dict[str, Mapping[str, int]] = {}
        frozen_fit_amplitudes: dict[str, np.ndarray] = {}
        frozen_query_amplitudes: dict[str, Mapping[str, np.ndarray]] = {}
        frozen_query_cosines: dict[str, Mapping[str, np.ndarray]] = {}
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
            frozen_counts[method] = _copy_confirmation_counts_antigravity(
                self.confirmation_counts[method],
                len(original),
                f"confirmation_counts[{method}]",
            )
            fit_amplitudes = _readonly_optional_vector_antigravity(
                self.fit_amplitudes[method],
                f"fit_amplitudes[{method}]",
            )
            query_amplitudes = _copy_query_evidence_antigravity(
                self.query_amplitudes[method],
                len(original),
                f"query_amplitudes[{method}]",
            )
            query_cosines = _copy_query_evidence_antigravity(
                self.query_cosines[method],
                len(original),
                f"query_cosines[{method}]",
            )
            if tuple(query_amplitudes) != tuple(query_cosines):
                raise ValueError("query amplitude/cosine scopes must match")
            if tuple(query_amplitudes) and len(fit_amplitudes) < 1:
                raise ValueError("evidence-bearing methods must keep fit amplitudes")
            if not tuple(query_amplitudes) and len(fit_amplitudes) != 0:
                raise ValueError("control methods must not keep fit amplitudes")
            frozen_fit_amplitudes[method] = fit_amplitudes
            frozen_query_amplitudes[method] = query_amplitudes
            frozen_query_cosines[method] = query_cosines

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
        object.__setattr__(self, "confirmation_counts", MappingProxyType(frozen_counts))
        object.__setattr__(self, "fit_amplitudes", MappingProxyType(frozen_fit_amplitudes))
        object.__setattr__(self, "query_amplitudes", MappingProxyType(frozen_query_amplitudes))
        object.__setattr__(self, "query_cosines", MappingProxyType(frozen_query_cosines))
        object.__setattr__(self, "feature_dimensions", MappingProxyType(frozen_dimensions))
        object.__setattr__(self, "provenance", MappingProxyType(frozen_provenance))


def _coerce_exp463_method_antigravity(
    value: object,
    query_len: int,
    expected_method: str,
    output_method: str,
) -> tuple[np.ndarray, Mapping[str, np.ndarray], int, str]:
    if not all(hasattr(value, attr) for attr in ("methods", "original", "primary")):
        raise ValueError("exp463 scorer returned an invalid result")
    methods = getattr(value, "methods")
    original = getattr(value, "original")
    primary = getattr(value, "primary")
    if type(methods) is not tuple or expected_method not in methods:
        raise ValueError(f"{output_method} control is missing from Exp463 replay")
    if not isinstance(original, Mapping) or not isinstance(primary, Mapping):
        raise ValueError("exp463 score containers are invalid")
    original_vector = _readonly_vector_antigravity(
        original[expected_method],
        f"{output_method} original",
    )
    if len(original_vector) != query_len:
        raise ValueError(f"{output_method} original length must match query size")
    primary_vectors = _copy_primary_antigravity(
        primary[expected_method],
        query_len,
        f"{output_method} primary",
    )
    dimensions = getattr(value, "feature_dimensions", {})
    dimension = dimensions.get(expected_method, 1) if isinstance(dimensions, Mapping) else 1
    if type(dimension) is not int or dimension < 1:
        raise ValueError(f"{output_method} feature dimension must be positive")
    provenance = getattr(value, "provenance", {})
    source = provenance.get(expected_method, output_method) if isinstance(provenance, Mapping) else output_method
    if type(source) is not str or not source:
        raise ValueError(f"{output_method} provenance must be non-empty")
    return original_vector, primary_vectors, dimension, source


def _centered_raw_evidence_antigravity(values: np.ndarray) -> tuple[float, float]:
    row = np.asarray(values, dtype=np.float64)
    suffix = row[FROZEN_CONFIG_ANTIGRAVITY.onset :]
    centered = suffix - float(np.mean(suffix))
    t = np.linspace(0.0, 1.0, len(suffix), dtype=np.float64)
    basis = t - float(np.mean(t))
    basis /= float(np.linalg.norm(basis))
    amplitude = float(np.dot(centered, basis))
    norm = float(np.linalg.norm(centered))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


def _uncentered_residual_evidence_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    residual = np.asarray(values, dtype=np.float64) - np.asarray(template, dtype=np.float64)
    suffix = residual[FROZEN_CONFIG_ANTIGRAVITY.onset :]
    t = np.linspace(0.0, 1.0, len(suffix), dtype=np.float64)
    basis = t / float(np.linalg.norm(t))
    amplitude = float(np.dot(suffix, basis))
    norm = float(np.linalg.norm(suffix))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


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


def _evidence_vectors_antigravity(
    rows: np.ndarray,
    evidence_function: Callable[[np.ndarray], object],
) -> tuple[np.ndarray, np.ndarray]:
    amplitudes: list[float] = []
    cosines: list[float] = []
    for row in rows:
        evidence = evidence_function(row)
        if hasattr(evidence, "amplitude") and hasattr(evidence, "cosine"):
            amplitude = getattr(evidence, "amplitude")
            cosine = getattr(evidence, "cosine")
        else:
            amplitude, cosine = evidence  # type: ignore[misc]
        amplitudes.append(float(amplitude))
        cosines.append(float(cosine))
    amp_array = np.asarray(amplitudes, dtype=np.float64)
    cos_array = np.asarray(cosines, dtype=np.float64)
    if not np.isfinite(amp_array).all() or not np.isfinite(cos_array).all():
        raise ValueError("evidence vectors must be finite")
    return amp_array, cos_array


def _confirmed_scores_antigravity(
    baseline: np.ndarray,
    fit_amplitudes: np.ndarray,
    query_amplitudes: np.ndarray,
    query_cosines: np.ndarray,
) -> tuple[np.ndarray, int]:
    ranks = anomaly_rank_antigravity(fit_amplitudes, query_amplitudes)
    confirmed = query_cosines >= FROZEN_CONFIG_ANTIGRAVITY.cosine_min
    scores = np.where(confirmed, np.maximum(np.asarray(baseline, dtype=np.float64), ranks), baseline)
    result = np.array(scores, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result, int(np.count_nonzero(confirmed))


def _always_max_scores_antigravity(
    baseline: np.ndarray,
    fit_amplitudes: np.ndarray,
    query_amplitudes: np.ndarray,
) -> np.ndarray:
    ranks = anomaly_rank_antigravity(fit_amplitudes, query_amplitudes)
    result = np.maximum(np.asarray(baseline, dtype=np.float64), ranks)
    output = np.array(result, dtype=np.float64, copy=True, order="C")
    output.setflags(write=False)
    return output


def _rank_only_scores_antigravity(
    fit_amplitudes: np.ndarray,
    query_amplitudes: np.ndarray,
) -> np.ndarray:
    return anomaly_rank_antigravity(fit_amplitudes, query_amplitudes)


def _zero_confirmation_counts_antigravity() -> Mapping[str, int]:
    return MappingProxyType({name: 0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY})


def _empty_query_evidence_antigravity() -> Mapping[str, np.ndarray]:
    return MappingProxyType({})


def _score_fold_methods_with_dependency_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: Exp463AuthorityAntigravity,
    exp463_fold_scorer: Callable[..., object],
) -> FoldMethodScoresAntigravity:
    trusted_authority = validate_exp463_authority_antigravity(authority)
    train_matrix = _train_antigravity(train)
    fit = _indices_antigravity(fit_indices, len(train_matrix), "fit_indices")
    query = _indices_antigravity(query_indices, len(train_matrix), "query_indices")
    if np.intersect1d(fit, query).size:
        raise ValueError("fit and query indices must be disjoint")
    seed_value = _seed_antigravity(seed)
    fold_value = _fold_index_antigravity(fold_index)
    del trusted_authority

    exp463_train = (
        train_matrix
        if train_matrix.dtype == np.dtype(np.float32)
        else np.asarray(train_matrix, dtype=np.float32)
    )
    exp463_scores = exp463_fold_scorer(
        exp463_train,
        fit,
        query,
        seed=seed_value,
        fold_index=fold_value,
    )
    c0_original, c0_primary, c0_dimension, c0_provenance = _coerce_exp463_method_antigravity(
        exp463_scores,
        len(query),
        "M0",
        "C0",
    )
    c1_original, c1_primary, c1_dimension, c1_provenance = _coerce_exp463_method_antigravity(
        exp463_scores,
        len(query),
        "M2",
        "C1",
    )
    c2_original, c2_primary, c2_dimension, c2_provenance = _coerce_exp463_method_antigravity(
        exp463_scores,
        len(query),
        "M4",
        "C2",
    )
    c7_original, c7_primary, c7_dimension, c7_provenance = _coerce_exp463_method_antigravity(
        exp463_scores,
        len(query),
        "M6",
        "C7",
    )

    template = np.median(np.asarray(train_matrix[fit], dtype=np.float64), axis=0)
    fit_rows = np.asarray(train_matrix[fit], dtype=np.float64)
    query_rows = np.asarray(train_matrix[query], dtype=np.float64)
    query_primary_rows = tuple(_primary_rows_antigravity(row) for row in query_rows)

    fit_c6_amp, _ = _evidence_vectors_antigravity(
        fit_rows,
        lambda row: centered_ramp_evidence_antigravity(row, template),
    )
    query_c6_amp, query_c6_cos = _evidence_vectors_antigravity(
        query_rows,
        lambda row: centered_ramp_evidence_antigravity(row, template),
    )
    c3_original = _rank_only_scores_antigravity(fit_c6_amp, query_c6_amp)
    c4_original = _always_max_scores_antigravity(c0_original, fit_c6_amp, query_c6_amp)
    c6_original, c6_normal_confirmed = _confirmed_scores_antigravity(
        c0_original,
        fit_c6_amp,
        query_c6_amp,
        query_c6_cos,
    )

    fit_c5_amp, _ = _evidence_vectors_antigravity(
        fit_rows,
        _centered_raw_evidence_antigravity,
    )
    query_c5_amp, query_c5_cos = _evidence_vectors_antigravity(
        query_rows,
        _centered_raw_evidence_antigravity,
    )
    c5_original, c5_normal_confirmed = _confirmed_scores_antigravity(
        c0_original,
        fit_c5_amp,
        query_c5_amp,
        query_c5_cos,
    )

    fit_c8_amp, _ = _evidence_vectors_antigravity(
        fit_rows,
        lambda row: _uncentered_residual_evidence_antigravity(row, template),
    )
    query_c8_amp, query_c8_cos = _evidence_vectors_antigravity(
        query_rows,
        lambda row: _uncentered_residual_evidence_antigravity(row, template),
    )
    c8_original, c8_normal_confirmed = _confirmed_scores_antigravity(
        c0_original,
        fit_c8_amp,
        query_c8_amp,
        query_c8_cos,
    )

    c3_primary: dict[str, np.ndarray] = {}
    c4_primary: dict[str, np.ndarray] = {}
    c5_primary: dict[str, np.ndarray] = {}
    c6_primary: dict[str, np.ndarray] = {}
    c8_primary: dict[str, np.ndarray] = {}
    c6_query_amplitudes: dict[str, np.ndarray] = {"normal": query_c6_amp}
    c6_query_cosines: dict[str, np.ndarray] = {"normal": query_c6_cos}
    c5_query_amplitudes: dict[str, np.ndarray] = {"normal": query_c5_amp}
    c5_query_cosines: dict[str, np.ndarray] = {"normal": query_c5_cos}
    c8_query_amplitudes: dict[str, np.ndarray] = {"normal": query_c8_amp}
    c8_query_cosines: dict[str, np.ndarray] = {"normal": query_c8_cos}
    c5_counts = {"normal": c5_normal_confirmed}
    c6_counts = {"normal": c6_normal_confirmed}
    c8_counts = {"normal": c8_normal_confirmed}

    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        rows = np.stack([row[change_type] for row in query_primary_rows])

        c6_amp, c6_cos = _evidence_vectors_antigravity(
            rows,
            lambda row: centered_ramp_evidence_antigravity(row, template),
        )
        c6_query_amplitudes[change_type] = c6_amp
        c6_query_cosines[change_type] = c6_cos
        c3_primary[change_type] = _rank_only_scores_antigravity(fit_c6_amp, c6_amp)
        c4_primary[change_type] = _always_max_scores_antigravity(
            c0_primary[change_type],
            fit_c6_amp,
            c6_amp,
        )
        c6_primary[change_type], c6_counts[change_type] = _confirmed_scores_antigravity(
            c0_primary[change_type],
            fit_c6_amp,
            c6_amp,
            c6_cos,
        )

        c5_amp, c5_cos = _evidence_vectors_antigravity(rows, _centered_raw_evidence_antigravity)
        c5_query_amplitudes[change_type] = c5_amp
        c5_query_cosines[change_type] = c5_cos
        c5_primary[change_type], c5_counts[change_type] = _confirmed_scores_antigravity(
            c0_primary[change_type],
            fit_c5_amp,
            c5_amp,
            c5_cos,
        )

        c8_amp, c8_cos = _evidence_vectors_antigravity(
            rows,
            lambda row: _uncentered_residual_evidence_antigravity(row, template),
        )
        c8_query_amplitudes[change_type] = c8_amp
        c8_query_cosines[change_type] = c8_cos
        c8_primary[change_type], c8_counts[change_type] = _confirmed_scores_antigravity(
            c0_primary[change_type],
            fit_c8_amp,
            c8_amp,
            c8_cos,
        )

    return FoldMethodScoresAntigravity(
        methods=_METHODS_ANTIGRAVITY,
        original={
            "C0": c0_original,
            "C1": c1_original,
            "C2": c2_original,
            "C3": c3_original,
            "C4": c4_original,
            "C5": c5_original,
            "C6": c6_original,
            "C7": c7_original,
            "C8": c8_original,
        },
        primary={
            "C0": c0_primary,
            "C1": c1_primary,
            "C2": c2_primary,
            "C3": MappingProxyType(c3_primary),
            "C4": MappingProxyType(c4_primary),
            "C5": MappingProxyType(c5_primary),
            "C6": MappingProxyType(c6_primary),
            "C7": c7_primary,
            "C8": MappingProxyType(c8_primary),
        },
        confirmation_counts={
            "C0": _zero_confirmation_counts_antigravity(),
            "C1": _zero_confirmation_counts_antigravity(),
            "C2": _zero_confirmation_counts_antigravity(),
            "C3": _zero_confirmation_counts_antigravity(),
            "C4": _zero_confirmation_counts_antigravity(),
            "C5": MappingProxyType(c5_counts),
            "C6": MappingProxyType(c6_counts),
            "C7": _zero_confirmation_counts_antigravity(),
            "C8": MappingProxyType(c8_counts),
        },
        fit_amplitudes={
            "C0": np.asarray([], dtype=np.float64),
            "C1": np.asarray([], dtype=np.float64),
            "C2": np.asarray([], dtype=np.float64),
            "C3": fit_c6_amp,
            "C4": fit_c6_amp,
            "C5": fit_c5_amp,
            "C6": fit_c6_amp,
            "C7": np.asarray([], dtype=np.float64),
            "C8": fit_c8_amp,
        },
        query_amplitudes={
            "C0": _empty_query_evidence_antigravity(),
            "C1": _empty_query_evidence_antigravity(),
            "C2": _empty_query_evidence_antigravity(),
            "C3": MappingProxyType(dict(c6_query_amplitudes)),
            "C4": MappingProxyType(dict(c6_query_amplitudes)),
            "C5": MappingProxyType(dict(c5_query_amplitudes)),
            "C6": MappingProxyType(dict(c6_query_amplitudes)),
            "C7": _empty_query_evidence_antigravity(),
            "C8": MappingProxyType(dict(c8_query_amplitudes)),
        },
        query_cosines={
            "C0": _empty_query_evidence_antigravity(),
            "C1": _empty_query_evidence_antigravity(),
            "C2": _empty_query_evidence_antigravity(),
            "C3": MappingProxyType(dict(c6_query_cosines)),
            "C4": MappingProxyType(dict(c6_query_cosines)),
            "C5": MappingProxyType(dict(c5_query_cosines)),
            "C6": MappingProxyType(dict(c6_query_cosines)),
            "C7": _empty_query_evidence_antigravity(),
            "C8": MappingProxyType(dict(c8_query_cosines)),
        },
        feature_dimensions={
            "C0": c0_dimension,
            "C1": c1_dimension,
            "C2": c2_dimension,
            "C3": 1,
            "C4": 2,
            "C5": 2,
            "C6": 2,
            "C7": c7_dimension,
            "C8": 2,
        },
        provenance={
            "C0": c0_provenance,
            "C1": c1_provenance,
            "C2": c2_provenance,
            "C3": "exp464_template_residual_centered_ramp_rank",
            "C4": "exp464_template_residual_centered_ramp_always_max",
            "C5": "exp464_raw_centered_ramp_confirmed_max",
            "C6": "exp464_template_residual_centered_ramp_confirmed_max",
            "C7": c7_provenance,
            "C8": "exp464_template_residual_uncentered_ramp_confirmed_max",
        },
    )


def score_fold_methods_antigravity(
    train: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: Exp463AuthorityAntigravity,
) -> FoldMethodScoresAntigravity:
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
        score_fold_methods_antigravity as score_exp463_fold_methods_antigravity,
    )

    return _score_fold_methods_with_dependency_antigravity(
        train,
        fit_indices,
        query_indices,
        seed=seed,
        fold_index=fold_index,
        authority=authority,
        exp463_fold_scorer=score_exp463_fold_methods_antigravity,
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "anomaly_rank_antigravity",
    "score_fold_methods_antigravity",
)
