"""Score-derived qualification for the Exp465 C0/C6/C8 boundary.

This module deliberately accepts already-produced fold scores.  It does not load
ECG200, open labels, or call a scorer.  The small amount of arithmetic here is
the evidence boundary used by the later result, replay, and dashboard modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any

import numpy as np

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
    mann_whitney_auc_antigravity,
    wilson_upper_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp465FoldResultContractAntigravity,
    Exp465QualificationContractAntigravity,
)


_METHODS_ANTIGRAVITY = ("C0", "C6", "C8")
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_CONFIRMATION_TYPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
_PAIR_TYPES_ANTIGRAVITY = ("gradual_drift", "period_change")
_PAIR_KEYS_ANTIGRAVITY = ("total_pairs", "fixed_pairs", "broken_pairs")
_WILSON_EXPLANATORY_ANTIGRAVITY = 0.05273725818905172
_MEAN_TOLERANCE_ANTIGRAVITY = 1e-12


def _exact_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _readonly_float_tuple_antigravity(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional vector")
        array = np.asarray(value, dtype=np.float64)
    elif type(value) in (tuple, list):
        if not value:
            raise ValueError(f"{name} must be non-empty")
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
    else:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return tuple(float(item) for item in array)


def _strict_int_antigravity(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_mapping_antigravity(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _ordered_float_mapping_antigravity(
    value: object,
    expected: tuple[str, ...],
    name: str,
    *,
    required: bool,
) -> Mapping[str, tuple[float, ...]]:
    mapping = _strict_mapping_antigravity(value, name)
    if not tuple(mapping):
        if required:
            raise ValueError(f"{name} cannot be empty")
        return MappingProxyType({})
    if tuple(mapping) != expected:
        raise ValueError(f"{name} must preserve exact order")
    frozen = {
        key: _readonly_float_tuple_antigravity(mapping[key], f"{name}[{key}]")
        for key in expected
    }
    return MappingProxyType(frozen)


def _ordered_count_mapping_antigravity(value: object, name: str) -> Mapping[str, int]:
    mapping = _strict_mapping_antigravity(value, name)
    if tuple(mapping) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve exact confirmation order")
    frozen = {
        key: _strict_int_antigravity(mapping[key], f"{name}[{key}]")
        for key in _CONFIRMATION_TYPES_ANTIGRAVITY
    }
    return MappingProxyType(frozen)


def _pair_mapping_antigravity(value: object) -> Mapping[str, Mapping[str, int]]:
    mapping = _strict_mapping_antigravity(value, "pair_diagnostics")
    if tuple(mapping) != _PAIR_TYPES_ANTIGRAVITY:
        raise ValueError("pair_diagnostics must preserve exact pair order")
    result: dict[str, Mapping[str, int]] = {}
    for change_type in _PAIR_TYPES_ANTIGRAVITY:
        stats = _strict_mapping_antigravity(mapping[change_type], f"pair_diagnostics[{change_type}]")
        if tuple(stats) != _PAIR_KEYS_ANTIGRAVITY:
            raise ValueError("pair diagnostics keys are invalid")
        total = _strict_int_antigravity(stats["total_pairs"], "total_pairs")
        fixed = _strict_int_antigravity(stats["fixed_pairs"], "fixed_pairs")
        broken = _strict_int_antigravity(stats["broken_pairs"], "broken_pairs")
        if fixed + broken > total:
            raise ValueError("fixed and broken pairs exceed total pairs")
        result[change_type] = MappingProxyType(
            {"total_pairs": total, "fixed_pairs": fixed, "broken_pairs": broken}
        )
    return MappingProxyType(result)


def _zero_pair_mapping_antigravity() -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            change_type: MappingProxyType(
                {key: 0 for key in _PAIR_KEYS_ANTIGRAVITY}
            )
            for change_type in _PAIR_TYPES_ANTIGRAVITY
        }
    )


def _metric_auc_antigravity(
    normal: tuple[float, ...],
    primary: Mapping[str, tuple[float, ...]],
) -> tuple[tuple[float, float, float, float], float, bool]:
    finite = bool(
        np.isfinite(np.asarray(normal, dtype=np.float64)).all()
        and all(
            np.isfinite(np.asarray(primary[name], dtype=np.float64)).all()
            for name in _CHANGE_TYPES_ANTIGRAVITY
        )
    )
    if not finite:
        return (0.0, 0.0, 0.0, 0.0), 0.0, False
    values = tuple(
        float(
            mann_whitney_auc_antigravity(
                np.asarray(normal, dtype=np.float64),
                np.asarray(primary[name], dtype=np.float64),
            )
        )
        for name in _CHANGE_TYPES_ANTIGRAVITY
    )
    return values, float(sum(values) / len(values)), True


def _confirmation_counts_from_score_antigravity(
    score: object,
    method: str,
    expected_length: int,
) -> Mapping[str, int]:
    raw_counts = _strict_mapping_antigravity(
        getattr(score, "confirmation_counts")[method],
        f"confirmation_counts[{method}]",
    )
    counts = _ordered_count_mapping_antigravity(raw_counts, f"confirmation_counts[{method}]")
    cosines = _strict_mapping_antigravity(
        getattr(score, "query_cosines")[method],
        f"query_cosines[{method}]",
    )
    if tuple(cosines) not in ((), _CONFIRMATION_TYPES_ANTIGRAVITY):
        raise ValueError(f"query_cosines[{method}] has an invalid scope")
    if tuple(cosines) == ():
        if any(counts[name] != 0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY):
            raise ValueError("a method without cosine evidence cannot confirm a row")
        return counts
    for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
        values = _readonly_float_tuple_antigravity(cosines[scope], f"query_cosines[{method}][{scope}]")
        if len(values) != expected_length:
            raise ValueError("query cosine length differs from score length")
        expected = sum(value >= FROZEN_CONFIG_ANTIGRAVITY.cosine_min for value in values)
        if counts[scope] != expected:
            raise ValueError("confirmation count differs from cosine evidence")
    return counts


def _query_cosines_antigravity(
    score: object,
    method: str,
    expected_length: int,
) -> Mapping[str, tuple[float, ...]]:
    raw = _strict_mapping_antigravity(
        getattr(score, "query_cosines")[method],
        f"query_cosines[{method}]",
    )
    if tuple(raw) == ():
        return MappingProxyType({})
    if tuple(raw) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError("query cosine scope order differs from frozen contract")
    return MappingProxyType(
        {
            scope: _readonly_float_tuple_antigravity(raw[scope], f"query_cosines[{method}][{scope}]")
            for scope in _CONFIRMATION_TYPES_ANTIGRAVITY
        }
    )


@dataclass(frozen=True, slots=True)
class Exp465MethodFoldMetricAntigravity:
    """One method/seed/fold record retaining raw scores and derived metrics."""

    method: str
    seed: int
    fold_index: int
    normal_scores: tuple[float, ...]
    primary_scores: Mapping[str, tuple[float, ...]]
    confirmation_counts: Mapping[str, int]
    query_cosines: Mapping[str, tuple[float, ...]]
    per_type_auc: tuple[float, float, float, float]
    mean_auc: float
    all_scores_finite: bool
    normal_wilson_upper: float
    pair_diagnostics: Mapping[str, Mapping[str, int]]
    passed: bool
    failure_reasons: tuple[str, ...]
    feature_dimension: int
    provenance: str

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _METHODS_ANTIGRAVITY:
            raise ValueError("method must be C0, C6, or C8")
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed is not frozen")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_index is not frozen")
        normal = _readonly_float_tuple_antigravity(self.normal_scores, "normal_scores")
        primary = _ordered_float_mapping_antigravity(
            self.primary_scores,
            _CHANGE_TYPES_ANTIGRAVITY,
            "primary_scores",
            required=True,
        )
        if any(len(primary[name]) != len(normal) for name in _CHANGE_TYPES_ANTIGRAVITY):
            raise ValueError("primary scores and normal scores must have equal lengths")
        counts = _ordered_count_mapping_antigravity(self.confirmation_counts, "confirmation_counts")
        if any(value > len(normal) for value in counts.values()):
            raise ValueError("confirmation count exceeds the normal score count")
        cosines = _ordered_float_mapping_antigravity(
            self.query_cosines,
            _CONFIRMATION_TYPES_ANTIGRAVITY,
            "query_cosines",
            required=False,
        )
        if cosines and any(len(cosines[name]) != len(normal) for name in _CONFIRMATION_TYPES_ANTIGRAVITY):
            raise ValueError("query cosine length differs from normal scores")
        auc, mean, finite = _metric_auc_antigravity(normal, primary)
        if tuple(self.per_type_auc) != auc:
            raise ValueError("per_type_auc differs from raw score replay")
        _exact_float_antigravity(self.mean_auc, "mean_auc")
        if abs(self.mean_auc - mean) > _MEAN_TOLERANCE_ANTIGRAVITY:
            raise ValueError("mean_auc differs from raw score replay")
        if type(self.all_scores_finite) is not bool or self.all_scores_finite is not finite:
            raise ValueError("all_scores_finite differs from raw score replay")
        expected_wilson = float(wilson_upper_antigravity(counts["normal"], len(normal)))
        _exact_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        if self.normal_wilson_upper != expected_wilson:
            raise ValueError("normal_wilson_upper differs from confirmation evidence")
        pair = _pair_mapping_antigravity(self.pair_diagnostics)
        if type(self.passed) is not bool:
            raise ValueError("passed must be bool")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be non-empty strings")
        if self.passed != (len(self.failure_reasons) == 0):
            raise ValueError("passed and failure_reasons disagree")
        if type(self.feature_dimension) is not int or self.feature_dimension < 1:
            raise ValueError("feature_dimension must be positive")
        if type(self.provenance) is not str or not self.provenance:
            raise ValueError("provenance must be non-empty")
        object.__setattr__(self, "normal_scores", normal)
        object.__setattr__(self, "primary_scores", primary)
        object.__setattr__(self, "confirmation_counts", counts)
        object.__setattr__(self, "query_cosines", cosines)
        object.__setattr__(self, "per_type_auc", auc)
        object.__setattr__(self, "mean_auc", mean)
        object.__setattr__(self, "all_scores_finite", finite)
        object.__setattr__(self, "normal_wilson_upper", expected_wilson)
        object.__setattr__(self, "pair_diagnostics", pair)


@dataclass(frozen=True, slots=True)
class Exp465MethodSummaryAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_per_type_auc: tuple[float, float, float, float]
    min_mean_auc: float
    max_normal_wilson_upper: float
    normal_confirmation_count: int
    all_scores_finite: bool
    pair_diagnostics: Mapping[str, Mapping[str, int]]
    passed: bool


@dataclass(frozen=True, slots=True)
class Exp465QualificationResultAntigravity(Exp465QualificationContractAntigravity):
    """Base Task 1 qualification contract plus replayable method evidence."""

    method_rows: tuple[Exp465MethodFoldMetricAntigravity, ...]
    summaries: tuple[Exp465MethodSummaryAntigravity, ...]

    def __post_init__(self) -> None:
        Exp465QualificationContractAntigravity.__post_init__(self)
        if type(self.method_rows) is not tuple or len(self.method_rows) != 30:
            raise ValueError("method_rows must contain 30 method/seed/fold rows")
        if any(type(row) is not Exp465MethodFoldMetricAntigravity for row in self.method_rows):
            raise ValueError("method_rows contain an invalid record")
        expected = tuple(
            (method, seed, fold)
            for method in _METHODS_ANTIGRAVITY
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        actual = tuple((row.method, row.seed, row.fold_index) for row in self.method_rows)
        if actual != expected:
            raise ValueError("method_rows must preserve method/seed/fold order")
        if type(self.summaries) is not tuple or tuple(item.method for item in self.summaries) != _METHODS_ANTIGRAVITY:
            raise ValueError("summaries must preserve method order")

    @property
    def contract(self) -> Exp465QualificationContractAntigravity:
        return Exp465QualificationContractAntigravity(
            rows=self.rows,
            seed_normal_confirmations=self.seed_normal_confirmations,
            label_query_count=self.label_query_count,
            official_evaluate_count=self.official_evaluate_count,
            artifact_map_sha256=self.artifact_map_sha256,
        )


def _normalise_fold_scores_antigravity(
    fold_scores: Mapping[tuple[int, int], object] | Iterable[object],
) -> tuple[tuple[int, int, object], ...]:
    if isinstance(fold_scores, Mapping):
        items = [(key, value) for key, value in fold_scores.items()]
    else:
        items = list(fold_scores)
    normalised: list[tuple[int, int, object]] = []
    if isinstance(fold_scores, Mapping):
        for key, score in items:
            if type(key) is not tuple or len(key) != 2:
                raise ValueError("fold score mapping keys must be (seed, fold_index)")
            normalised.append((key[0], key[1], score))
    else:
        for item in items:
            if type(item) is tuple and len(item) == 3:
                normalised.append((item[0], item[1], item[2]))
            else:
                raise ValueError("fold score iterable items must be (seed, fold_index, score)")
    expected = tuple(
        (seed, fold)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    actual = tuple((seed, fold) for seed, fold, _ in normalised)
    if actual != expected:
        raise ValueError("fold scores must contain the exact frozen seed/fold order")
    return tuple(normalised)


def _metric_from_score_antigravity(
    score: object,
    method: str,
    seed: int,
    fold_index: int,
) -> Exp465MethodFoldMetricAntigravity:
    if tuple(getattr(score, "methods")) != _METHODS_ANTIGRAVITY:
        raise ValueError("score methods must preserve C0/C6/C8 order")
    normal = _readonly_float_tuple_antigravity(getattr(score, "original")[method], f"original[{method}]")
    primary = _ordered_float_mapping_antigravity(
        getattr(score, "primary")[method],
        _CHANGE_TYPES_ANTIGRAVITY,
        f"primary[{method}]",
        required=True,
    )
    if any(len(primary[name]) != len(normal) for name in _CHANGE_TYPES_ANTIGRAVITY):
        raise ValueError("primary and normal score lengths differ")
    counts = _confirmation_counts_from_score_antigravity(score, method, len(normal))
    cosines = _query_cosines_antigravity(score, method, len(normal))
    auc, mean, finite = _metric_auc_antigravity(normal, primary)
    reasons = _base_failure_reasons_antigravity(auc, mean, finite)
    return Exp465MethodFoldMetricAntigravity(
        method=method,
        seed=seed,
        fold_index=fold_index,
        normal_scores=normal,
        primary_scores=primary,
        confirmation_counts=counts,
        query_cosines=cosines,
        per_type_auc=auc,
        mean_auc=mean,
        all_scores_finite=finite,
        normal_wilson_upper=float(wilson_upper_antigravity(counts["normal"], len(normal))),
        pair_diagnostics=_zero_pair_mapping_antigravity(),
        passed=not reasons,
        failure_reasons=tuple(reasons),
        feature_dimension=int(getattr(score, "feature_dimensions")[method]),
        provenance=str(getattr(score, "provenance")[method]),
    )


def _base_failure_reasons_antigravity(
    auc: tuple[float, float, float, float],
    mean: float,
    finite: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not finite:
        reasons.append("nonfinite_scores")
    for change_type, value in zip(_CHANGE_TYPES_ANTIGRAVITY, auc, strict=True):
        if value < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"{change_type}_auc_below_0.85")
    if mean < FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min:
        reasons.append("mean_auc_below_0.90")
    return tuple(reasons)


def _replay_base_row_antigravity(
    row: Exp465MethodFoldMetricAntigravity,
) -> Exp465MethodFoldMetricAntigravity:
    """Rebuild every score-derived row field before C8 comparison is applied."""
    counts = _ordered_count_mapping_antigravity(row.confirmation_counts, "confirmation_counts")
    cosines = _ordered_float_mapping_antigravity(
        row.query_cosines, _CONFIRMATION_TYPES_ANTIGRAVITY, "query_cosines", required=False,
    )
    if cosines:
        for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
            if sum(value >= FROZEN_CONFIG_ANTIGRAVITY.cosine_min for value in cosines[scope]) != counts[scope]:
                raise ValueError("confirmation count differs from cosine evidence")
    elif any(counts.values()):
        raise ValueError("a method without cosine evidence cannot confirm a row")
    auc, mean, finite = _metric_auc_antigravity(row.normal_scores, row.primary_scores)
    return Exp465MethodFoldMetricAntigravity(
        method=row.method, seed=row.seed, fold_index=row.fold_index,
        normal_scores=row.normal_scores, primary_scores=row.primary_scores,
        confirmation_counts=counts, query_cosines=cosines, per_type_auc=auc,
        mean_auc=mean, all_scores_finite=finite,
        normal_wilson_upper=float(wilson_upper_antigravity(counts["normal"], len(row.normal_scores))),
        pair_diagnostics=_zero_pair_mapping_antigravity(),
        passed=not _base_failure_reasons_antigravity(auc, mean, finite),
        failure_reasons=_base_failure_reasons_antigravity(auc, mean, finite),
        feature_dimension=row.feature_dimension, provenance=row.provenance,
    )


def _pair_diagnostics_for_c8_antigravity(
    c0: Exp465MethodFoldMetricAntigravity,
    c8: Exp465MethodFoldMetricAntigravity,
) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for change_type in _PAIR_TYPES_ANTIGRAVITY:
        c0_changes = c0.primary_scores[change_type]
        c8_changes = c8.primary_scores[change_type]
        total = len(c0.normal_scores) * len(c0_changes)
        fixed = 0
        broken = 0
        for normal_index, c0_normal in enumerate(c0.normal_scores):
            c8_normal = c8.normal_scores[normal_index]
            for change_index, c0_change in enumerate(c0_changes):
                c8_change = c8_changes[change_index]
                if c0_change == c0_normal or c8_change == c8_normal:
                    continue
                correct_c0 = c0_change > c0_normal
                correct_c8 = c8_change > c8_normal
                if not correct_c0 and correct_c8:
                    fixed += 1
                elif correct_c0 and not correct_c8:
                    broken += 1
        result[change_type] = MappingProxyType(
            {"total_pairs": total, "fixed_pairs": fixed, "broken_pairs": broken}
        )
    return MappingProxyType(result)


def _with_c8_comparison_antigravity(
    c8: Exp465MethodFoldMetricAntigravity,
    c0: Exp465MethodFoldMetricAntigravity,
) -> Exp465MethodFoldMetricAntigravity:
    reasons = list(c8.failure_reasons)
    if c8.per_type_auc[1] < c0.per_type_auc[1]:
        reasons.append("gradual_auc_below_c0")
    if c8.per_type_auc[3] < c0.per_type_auc[3]:
        reasons.append("period_auc_below_c0")
    return Exp465MethodFoldMetricAntigravity(
        method=c8.method,
        seed=c8.seed,
        fold_index=c8.fold_index,
        normal_scores=c8.normal_scores,
        primary_scores=c8.primary_scores,
        confirmation_counts=c8.confirmation_counts,
        query_cosines=c8.query_cosines,
        per_type_auc=c8.per_type_auc,
        mean_auc=c8.mean_auc,
        all_scores_finite=c8.all_scores_finite,
        normal_wilson_upper=c8.normal_wilson_upper,
        pair_diagnostics=_pair_diagnostics_for_c8_antigravity(c0, c8),
        passed=not reasons,
        failure_reasons=tuple(sorted(set(reasons))),
        feature_dimension=c8.feature_dimension,
        provenance=c8.provenance,
    )


def _summary_antigravity(
    rows: tuple[Exp465MethodFoldMetricAntigravity, ...],
    method: str,
) -> Exp465MethodSummaryAntigravity:
    method_rows = tuple(row for row in rows if row.method == method)
    if len(method_rows) != FROZEN_CONFIG_ANTIGRAVITY.stability_record_count:
        raise ValueError("each method must contain ten rows")
    pair = {
        change_type: {
            key: sum(row.pair_diagnostics[change_type][key] for row in method_rows)
            for key in _PAIR_KEYS_ANTIGRAVITY
        }
        for change_type in _PAIR_TYPES_ANTIGRAVITY
    }
    return Exp465MethodSummaryAntigravity(
        method=method,
        fold_count=len(method_rows),
        passed_count=sum(row.passed for row in method_rows),
        failed_count=sum(not row.passed for row in method_rows),
        min_per_type_auc=tuple(
            min(row.per_type_auc[index] for row in method_rows)
            for index in range(len(_CHANGE_TYPES_ANTIGRAVITY))
        ),
        min_mean_auc=min(row.mean_auc for row in method_rows),
        max_normal_wilson_upper=max(row.normal_wilson_upper for row in method_rows),
        normal_confirmation_count=sum(
            row.confirmation_counts["normal"] for row in method_rows
        ),
        all_scores_finite=all(row.all_scores_finite for row in method_rows),
        pair_diagnostics=MappingProxyType(
            {
                change_type: MappingProxyType(values)
                for change_type, values in pair.items()
            }
        ),
        passed=all(row.passed for row in method_rows),
    )


def qualify_methods_antigravity(
    fold_scores: Mapping[tuple[int, int], object] | Iterable[object],
) -> Exp465QualificationResultAntigravity:
    """Recompute all Task 4 metrics from exactly ten Task 3 score records."""

    records = _normalise_fold_scores_antigravity(fold_scores)
    method_rows: list[Exp465MethodFoldMetricAntigravity] = []
    for seed, fold_index, score in records:
        for method in _METHODS_ANTIGRAVITY:
            method_rows.append(_metric_from_score_antigravity(score, method, seed, fold_index))
    by_identity = {(row.method, row.seed, row.fold_index): row for row in method_rows}
    updated: list[Exp465MethodFoldMetricAntigravity] = []
    for method in _METHODS_ANTIGRAVITY:
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds):
                row = by_identity[(method, seed, fold_index)]
                if method == "C8":
                    row = _with_c8_comparison_antigravity(
                        row,
                        by_identity[("C0", seed, fold_index)],
                    )
                updated.append(row)
    ordered_rows = tuple(updated)
    c8_rows = tuple(row for row in ordered_rows if row.method == "C8")
    result_rows = tuple(
        Exp465FoldResultContractAntigravity(
            seed=row.seed,
            fold_index=row.fold_index,
            per_type_auc=row.per_type_auc,
            mean_auc=row.mean_auc,
            c0_gradual_auc=by_identity[("C0", row.seed, row.fold_index)].per_type_auc[1],
            c0_period_auc=by_identity[("C0", row.seed, row.fold_index)].per_type_auc[3],
            c8_gradual_auc=row.per_type_auc[1],
            c8_period_auc=row.per_type_auc[3],
            all_scores_finite=row.all_scores_finite,
            passed=row.passed,
            failure_reasons=row.failure_reasons,
        )
        for row in c8_rows
    )
    seed_confirmations = MappingProxyType(
        {
            seed: sum(
                by_identity[("C8", seed, fold)].confirmation_counts["normal"]
                for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
            )
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        }
    )
    summaries = tuple(_summary_antigravity(ordered_rows, method) for method in _METHODS_ANTIGRAVITY)
    return Exp465QualificationResultAntigravity(
        rows=result_rows,
        seed_normal_confirmations=seed_confirmations,
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        official_evaluate_count=FROZEN_CONFIG_ANTIGRAVITY.official_evaluate_count_must_remain,
        artifact_map_sha256=None,
        method_rows=ordered_rows,
        summaries=summaries,
    )


def recompute_qualification_antigravity(
    qualification: Exp465QualificationResultAntigravity,
) -> Exp465QualificationResultAntigravity:
    """Rebuild the result from its stored raw score arrays for replay."""

    if type(qualification) is not Exp465QualificationResultAntigravity:
        raise ValueError("semantic replay requires the Exp465 qualification result")
    stored_rows = qualification.method_rows
    base_rows = tuple(_replay_base_row_antigravity(row) for row in stored_rows)
    by_identity = {(row.method, row.seed, row.fold_index): row for row in base_rows}
    rows = tuple(
        _with_c8_comparison_antigravity(row, by_identity[("C0", row.seed, row.fold_index)]) if row.method == "C8" else row
        for row in base_rows
    )
    c8_rows = tuple(row for row in rows if row.method == "C8")
    expected_c8 = tuple(
        _with_c8_comparison_antigravity(
            row, next(item for item in rows if item.method == "C0" and item.seed == row.seed and item.fold_index == row.fold_index),
        )
        for row in c8_rows
    )
    if stored_rows != rows:
        raise ValueError("method rows differ from raw score replay")
    if tuple(qualification.rows) != tuple(
        Exp465FoldResultContractAntigravity(
            seed=row.seed,
            fold_index=row.fold_index,
            per_type_auc=row.per_type_auc,
            mean_auc=row.mean_auc,
            c0_gradual_auc=next(item for item in rows if item.method == "C0" and item.seed == row.seed and item.fold_index == row.fold_index).per_type_auc[1],
            c0_period_auc=next(item for item in rows if item.method == "C0" and item.seed == row.seed and item.fold_index == row.fold_index).per_type_auc[3],
            c8_gradual_auc=row.per_type_auc[1],
            c8_period_auc=row.per_type_auc[3],
            all_scores_finite=row.all_scores_finite,
            passed=row.passed,
            failure_reasons=row.failure_reasons,
        )
        for row in expected_c8
    ):
        raise ValueError("qualification rows differ from raw score replay")
    expected_seed = MappingProxyType(
        {
            seed: sum(
                row.confirmation_counts["normal"]
                for row in rows
                if row.method == "C8" and row.seed == seed
            )
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        }
    )
    if dict(qualification.seed_normal_confirmations) != dict(expected_seed):
        raise ValueError("seed confirmation summary differs from raw evidence")
    expected_summaries = tuple(_summary_antigravity(rows, method) for method in _METHODS_ANTIGRAVITY)
    if qualification.summaries != expected_summaries:
        raise ValueError("method summaries differ from raw evidence")
    return qualification


__all__ = (
    "Exp465MethodFoldMetricAntigravity",
    "Exp465MethodSummaryAntigravity",
    "Exp465QualificationResultAntigravity",
    "_CHANGE_TYPES_ANTIGRAVITY",
    "_CONFIRMATION_TYPES_ANTIGRAVITY",
    "_METHODS_ANTIGRAVITY",
    "_PAIR_TYPES_ANTIGRAVITY",
    "_WILSON_EXPLANATORY_ANTIGRAVITY",
    "qualify_methods_antigravity",
    "recompute_qualification_antigravity",
)
