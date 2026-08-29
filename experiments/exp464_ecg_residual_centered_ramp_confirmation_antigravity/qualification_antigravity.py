"""Exp464 Task 4 qualification contracts and score-derived metrics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from math import isfinite
from types import MappingProxyType
import json

import numpy as np

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
    blocked_plan_antigravity,
)
from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
    mann_whitney_auc_antigravity,
    wilson_upper_antigravity,
)

from .config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
    HistoricalPeriodThresholdAntigravity,
)
from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)
from .raw_evidence_antigravity import (
    amplitudes_cosines_from_vectors_antigravity,
    decode_vector_matrix_antigravity,
)
from .scoring_antigravity import anomaly_rank_antigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_CONFIRMATION_TYPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
_PAIR_TYPES_ANTIGRAVITY = ("gradual_drift", "period_change")
_GATE_ANTIGRAVITY = 1.0
_EVIDENCE_METHODS_ANTIGRAVITY = ("C3", "C4", "C5", "C6", "C8")
_RAW_CARRIER_METHODS_ANTIGRAVITY = ("C5", "C6", "C8")


def _int_antigravity(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _score_tuple_antigravity(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1 or value.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional score vector")
        result = tuple(float(item) for item in np.asarray(value, dtype=np.float64))
    elif type(value) in (tuple, list):
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
    else:
        raise ValueError(f"{name} must be a score vector")
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _optional_score_tuple_antigravity(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional score vector")
        result = tuple(float(item) for item in np.asarray(value, dtype=np.float64))
    elif type(value) in (tuple, list):
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
    else:
        raise ValueError(f"{name} must be a score vector")
    return result


def _score_array_antigravity(value: tuple[float, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional score vector")
    return array


def _per_type_float_mapping_antigravity(
    value: Mapping[str, object],
    name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve exact change-type order")
    return MappingProxyType(
        {
            change_type: _unit_float_antigravity(
                value[change_type],
                f"{name}[{change_type}]",
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )


def _primary_scores_antigravity(
    value: Mapping[str, object],
    expected_len: int,
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError("primary_scores must preserve exact change-type order")
    frozen: dict[str, tuple[float, ...]] = {}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        scores = _score_tuple_antigravity(
            value[change_type],
            f"primary_scores[{change_type}]",
        )
        if len(scores) != expected_len:
            raise ValueError("primary score lengths must equal normal score length")
        frozen[change_type] = scores
    return MappingProxyType(frozen)


def _confirmation_counts_antigravity(
    value: Mapping[str, object],
    normal_trials: int,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or tuple(value) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError("confirmation_counts must preserve exact confirmation order")
    frozen: dict[str, int] = {}
    for name in _CONFIRMATION_TYPES_ANTIGRAVITY:
        frozen[name] = _int_antigravity(value[name], f"confirmation_counts[{name}]")
        if frozen[name] > normal_trials:
            raise ValueError("confirmation count cannot exceed normal_trials")
    return MappingProxyType(frozen)


def _confirmation_rates_antigravity(
    value: Mapping[str, object],
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError("confirmation_rates must preserve exact confirmation order")
    return MappingProxyType(
        {
            name: _unit_float_antigravity(
                value[name],
                f"confirmation_rates[{name}]",
            )
            for name in _CONFIRMATION_TYPES_ANTIGRAVITY
        }
    )


def _query_evidence_mapping_antigravity(
    value: Mapping[str, object],
    expected_len: int,
    name: str,
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(value) == ():
        return MappingProxyType({})
    if tuple(value) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve exact evidence scope order")
    frozen: dict[str, tuple[float, ...]] = {}
    for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
        scores = _score_tuple_antigravity(value[scope], f"{name}[{scope}]")
        if len(scores) != expected_len:
            raise ValueError(f"{name}[{scope}] length must equal normal_trials")
        frozen[scope] = scores
    return MappingProxyType(frozen)


def _query_blob_mapping_antigravity(
    value: Mapping[str, object],
    name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(value) == ():
        return MappingProxyType({})
    if tuple(value) != _CONFIRMATION_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve exact evidence scope order")
    frozen: dict[str, str] = {}
    for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
        blob = value[scope]
        if type(blob) is not str or not blob:
            raise ValueError(f"{name}[{scope}] must be a non-empty string")
        frozen[scope] = blob
    return MappingProxyType(frozen)


def _pair_stat_antigravity(value: Mapping[str, object], name: str) -> Mapping[str, int]:
    expected = ("total_pairs", "fixed_pairs", "broken_pairs")
    if not isinstance(value, Mapping) or tuple(value) != expected:
        raise ValueError(f"{name} must preserve exact pair-stat order")
    total_pairs = _int_antigravity(value["total_pairs"], f"{name}[total_pairs]")
    fixed_pairs = _int_antigravity(value["fixed_pairs"], f"{name}[fixed_pairs]")
    broken_pairs = _int_antigravity(value["broken_pairs"], f"{name}[broken_pairs]")
    if fixed_pairs + broken_pairs > total_pairs:
        raise ValueError(f"{name} cannot count beyond total_pairs")
    return MappingProxyType(
        {
            "total_pairs": total_pairs,
            "fixed_pairs": fixed_pairs,
            "broken_pairs": broken_pairs,
        }
    )


def _pair_diagnostics_antigravity(
    value: Mapping[str, object],
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(value, Mapping) or tuple(value) != _PAIR_TYPES_ANTIGRAVITY:
        raise ValueError("pair_diagnostics must preserve exact pair-type order")
    return MappingProxyType(
        {
            change_type: _pair_stat_antigravity(
                value[change_type], f"pair_diagnostics[{change_type}]"
            )
            for change_type in _PAIR_TYPES_ANTIGRAVITY
        }
    )


def _base_row_failure_reasons_antigravity(
    *,
    per_type_auc: Mapping[str, float],
    mean_auc: float,
    normal_wilson_upper: float,
    scores_finite: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not scores_finite:
        reasons.append("nonfinite_scores")
    if normal_wilson_upper > FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max:
        reasons.append("normal_wilson_upper_above_max")
    if mean_auc < FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min:
        reasons.append("mean_auc_below_min")
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        if per_type_auc[change_type] < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"{change_type}_auc_below_min")
    return tuple(reasons)


def _confirmation_rates_from_counts_antigravity(
    counts: Mapping[str, int],
    normal_trials: int,
) -> Mapping[str, float]:
    return MappingProxyType(
        {
            name: float(counts[name] / normal_trials)
            for name in _CONFIRMATION_TYPES_ANTIGRAVITY
        }
    )


def _zero_pair_diagnostics_antigravity() -> Mapping[str, Mapping[str, int]]:
    return MappingProxyType(
        {
            change_type: MappingProxyType(
                {"total_pairs": 0, "fixed_pairs": 0, "broken_pairs": 0}
            )
            for change_type in _PAIR_TYPES_ANTIGRAVITY
        }
    )


def _empty_evidence_mapping_antigravity() -> Mapping[str, tuple[float, ...]]:
    return MappingProxyType({})


def _safe_metrics_antigravity(
    normal: np.ndarray,
    primary: Mapping[str, np.ndarray],
) -> tuple[int, float, Mapping[str, float], float, bool]:
    finite = bool(
        np.isfinite(normal).all()
        and all(np.isfinite(primary[change_type]).all() for change_type in _CHANGE_TYPES_ANTIGRAVITY)
    )
    if not finite:
        zeros = MappingProxyType({change_type: 0.0 for change_type in _CHANGE_TYPES_ANTIGRAVITY})
        return 0, 1.0, zeros, 0.0, False
    alerts = int(np.count_nonzero(normal > _GATE_ANTIGRAVITY))
    per_type = MappingProxyType(
        {
            change_type: float(
                mann_whitney_auc_antigravity(normal, primary[change_type])
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )
    mean_auc = float(sum(per_type.values()) / len(_CHANGE_TYPES_ANTIGRAVITY))
    return alerts, float(wilson_upper_antigravity(alerts, len(normal))), per_type, mean_auc, True


def _strict_row_identity_antigravity(row: "Exp464MethodFoldMetricAntigravity") -> tuple[str, int, int]:
    return (row.method, row.seed, row.fold_index)


def _expected_row_identities_antigravity() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (method, seed, fold_index)
        for method in _METHODS_ANTIGRAVITY
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )


@dataclass(frozen=True, slots=True)
class Exp464MethodFoldMetricAntigravity:
    method: str
    seed: int
    fold_index: int
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    per_type_auc: Mapping[str, float]
    mean_auc: float
    scores_finite: bool
    normal_scores: tuple[float, ...]
    primary_scores: Mapping[str, tuple[float, ...]]
    confirmation_counts: Mapping[str, int]
    confirmation_rates: Mapping[str, float]
    pair_diagnostics: Mapping[str, Mapping[str, int]]
    passed: bool
    failure_reasons: tuple[str, ...]
    feature_dimension: int
    provenance: str
    fit_amplitudes: tuple[float, ...]
    query_amplitudes: Mapping[str, tuple[float, ...]]
    query_cosines: Mapping[str, tuple[float, ...]]
    raw_evidence_kind: str
    raw_evidence_source_method: str
    raw_fit_vector_blob: str
    raw_query_vector_blobs: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _METHODS_ANTIGRAVITY:
            raise ValueError("method must be one of frozen C0-C8 IDs")
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed must be one of the frozen seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_index must be one of the frozen blocked folds")
        normal_trials = _int_antigravity(self.normal_trials, "normal_trials", minimum=1)
        _int_antigravity(self.normal_alerts, "normal_alerts")
        _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        object.__setattr__(self, "per_type_auc", _per_type_float_mapping_antigravity(self.per_type_auc, "per_type_auc"))
        _unit_float_antigravity(self.mean_auc, "mean_auc")
        _bool_antigravity(self.scores_finite, "scores_finite")
        normal_scores = _score_tuple_antigravity(self.normal_scores, "normal_scores")
        if len(normal_scores) != normal_trials:
            raise ValueError("normal_scores length must equal normal_trials")
        object.__setattr__(self, "normal_scores", normal_scores)
        object.__setattr__(
            self,
            "primary_scores",
            _primary_scores_antigravity(self.primary_scores, normal_trials),
        )
        object.__setattr__(
            self,
            "confirmation_counts",
            _confirmation_counts_antigravity(self.confirmation_counts, normal_trials),
        )
        object.__setattr__(
            self,
            "confirmation_rates",
            _confirmation_rates_antigravity(self.confirmation_rates),
        )
        object.__setattr__(
            self,
            "pair_diagnostics",
            _pair_diagnostics_antigravity(self.pair_diagnostics),
        )
        _bool_antigravity(self.passed, "passed")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be a tuple of non-empty strings")
        _int_antigravity(self.feature_dimension, "feature_dimension", minimum=1)
        if type(self.provenance) is not str or not self.provenance:
            raise ValueError("provenance must be a non-empty string")
        fit_amplitudes = _optional_score_tuple_antigravity(self.fit_amplitudes, "fit_amplitudes")
        object.__setattr__(self, "fit_amplitudes", fit_amplitudes)
        object.__setattr__(
            self,
            "query_amplitudes",
            _query_evidence_mapping_antigravity(
                self.query_amplitudes,
                normal_trials,
                "query_amplitudes",
            ),
        )
        object.__setattr__(
            self,
            "query_cosines",
            _query_evidence_mapping_antigravity(
                self.query_cosines,
                normal_trials,
                "query_cosines",
            ),
        )
        if tuple(self.query_amplitudes) != tuple(self.query_cosines):
            raise ValueError("query amplitude/cosine scope order must match")
        has_evidence = bool(tuple(self.query_amplitudes))
        if self.method in _EVIDENCE_METHODS_ANTIGRAVITY:
            if not has_evidence or len(fit_amplitudes) < 1:
                raise ValueError("evidence-bearing methods must keep fit amplitudes and query evidence")
        else:
            if has_evidence or fit_amplitudes:
                raise ValueError("control methods must not keep evidence vectors")
        if type(self.raw_evidence_kind) is not str:
            raise ValueError("raw_evidence_kind must be a string")
        if type(self.raw_evidence_source_method) is not str:
            raise ValueError("raw_evidence_source_method must be a string")
        if type(self.raw_fit_vector_blob) is not str:
            raise ValueError("raw_fit_vector_blob must be a string")
        object.__setattr__(
            self,
            "raw_query_vector_blobs",
            _query_blob_mapping_antigravity(
                self.raw_query_vector_blobs,
                "raw_query_vector_blobs",
            ),
        )
        if self.method in _RAW_CARRIER_METHODS_ANTIGRAVITY:
            if self.raw_evidence_kind not in (
                "raw_centered",
                "template_residual_centered",
                "template_residual_uncentered",
            ):
                raise ValueError("carrier rows must declare a valid raw_evidence_kind")
            if self.raw_evidence_source_method != self.method:
                raise ValueError("carrier rows must source their own raw evidence")
            if not self.raw_fit_vector_blob or tuple(self.raw_query_vector_blobs) != _CONFIRMATION_TYPES_ANTIGRAVITY:
                raise ValueError("carrier rows must keep raw fit/query vector blobs")
        elif self.method in ("C3", "C4"):
            if (
                self.raw_evidence_kind
                or self.raw_evidence_source_method != "C6"
                or self.raw_fit_vector_blob
                or tuple(self.raw_query_vector_blobs)
            ):
                raise ValueError("C3/C4 rows must reference C6 raw evidence without blobs")
        else:
            if (
                self.raw_evidence_kind
                or self.raw_evidence_source_method
                or self.raw_fit_vector_blob
                or tuple(self.raw_query_vector_blobs)
            ):
                raise ValueError("control rows must not keep raw evidence metadata")


@dataclass(frozen=True, slots=True)
class Exp464MethodSummaryAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_per_type_auc: Mapping[str, float]
    min_gradual_drift_auc: float
    min_period_change_auc: float
    min_mean_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool
    passed: bool

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _METHODS_ANTIGRAVITY:
            raise ValueError("summary method must be one of frozen C0-C8 IDs")
        fold_count = _int_antigravity(self.fold_count, "fold_count", minimum=1)
        passed_count = _int_antigravity(self.passed_count, "passed_count")
        failed_count = _int_antigravity(self.failed_count, "failed_count")
        if fold_count != passed_count + failed_count:
            raise ValueError("summary counts must cover all folds")
        object.__setattr__(
            self,
            "min_per_type_auc",
            _per_type_float_mapping_antigravity(self.min_per_type_auc, "min_per_type_auc"),
        )
        _unit_float_antigravity(self.min_gradual_drift_auc, "min_gradual_drift_auc")
        _unit_float_antigravity(self.min_period_change_auc, "min_period_change_auc")
        _unit_float_antigravity(self.min_mean_auc, "min_mean_auc")
        _unit_float_antigravity(self.max_normal_wilson_upper, "max_normal_wilson_upper")
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class Exp464QualificationContractAntigravity:
    dataset: str
    method_order: tuple[str, ...]
    primary_candidate: str
    fold_metrics: tuple[Exp464MethodFoldMetricAntigravity, ...]
    summaries: tuple[Exp464MethodSummaryAntigravity, ...]
    label_query_count: int

    def __post_init__(self) -> None:
        if self.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
            raise ValueError("dataset differs from the frozen Exp464 contract")
        if type(self.method_order) is not tuple or self.method_order != _METHODS_ANTIGRAVITY:
            raise ValueError("method_order must preserve exact C0-C8 order")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary_candidate differs from the frozen Exp464 contract")
        if type(self.fold_metrics) is not tuple or len(self.fold_metrics) != len(_expected_row_identities_antigravity()):
            raise ValueError("fold_metrics must contain exact 90 method/seed/fold rows")
        if any(type(row) is not Exp464MethodFoldMetricAntigravity for row in self.fold_metrics):
            raise ValueError("fold_metrics must contain exact Exp464MethodFoldMetricAntigravity values")
        if tuple(_strict_row_identity_antigravity(row) for row in self.fold_metrics) != _expected_row_identities_antigravity():
            raise ValueError("fold_metrics order must match method/seed/fold order")
        if type(self.summaries) is not tuple or len(self.summaries) != len(_METHODS_ANTIGRAVITY):
            raise ValueError("summaries must contain exact nine method summaries")
        if any(type(summary) is not Exp464MethodSummaryAntigravity for summary in self.summaries):
            raise ValueError("summaries must contain exact Exp464MethodSummaryAntigravity values")
        if tuple(summary.method for summary in self.summaries) != _METHODS_ANTIGRAVITY:
            raise ValueError("summaries must preserve exact C0-C8 order")
        if type(self.label_query_count) is not int or self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("label_query_count must remain zero")


def recompute_metric_from_scores_antigravity(
    row: Exp464MethodFoldMetricAntigravity,
) -> Exp464MethodFoldMetricAntigravity:
    normal = _score_array_antigravity(row.normal_scores, "normal_scores")
    primary = {
        change_type: _score_array_antigravity(
            row.primary_scores[change_type],
            f"primary_scores[{change_type}]",
        )
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    normal_alerts, normal_wilson_upper, per_type_auc, mean_auc, scores_finite = _safe_metrics_antigravity(
        normal,
        primary,
    )
    confirmation_rates = _confirmation_rates_from_counts_antigravity(
        row.confirmation_counts,
        len(normal),
    )
    failure_reasons = _base_row_failure_reasons_antigravity(
        per_type_auc=per_type_auc,
        mean_auc=mean_auc,
        normal_wilson_upper=normal_wilson_upper,
        scores_finite=scores_finite,
    )
    return Exp464MethodFoldMetricAntigravity(
        method=row.method,
        seed=row.seed,
        fold_index=row.fold_index,
        normal_trials=len(normal),
        normal_alerts=normal_alerts,
        normal_wilson_upper=normal_wilson_upper,
        per_type_auc=per_type_auc,
        mean_auc=mean_auc,
        scores_finite=scores_finite,
        normal_scores=row.normal_scores,
        primary_scores=row.primary_scores,
        confirmation_counts=row.confirmation_counts,
        confirmation_rates=confirmation_rates,
        pair_diagnostics=row.pair_diagnostics,
        passed=(not failure_reasons),
        failure_reasons=failure_reasons,
        feature_dimension=row.feature_dimension,
        provenance=row.provenance,
        fit_amplitudes=row.fit_amplitudes,
        query_amplitudes=row.query_amplitudes,
        query_cosines=row.query_cosines,
        raw_evidence_kind=row.raw_evidence_kind,
        raw_evidence_source_method=row.raw_evidence_source_method,
        raw_fit_vector_blob=row.raw_fit_vector_blob,
        raw_query_vector_blobs=row.raw_query_vector_blobs,
    )


def _pair_diagnostics_for_rows_antigravity(
    c0_row: Exp464MethodFoldMetricAntigravity,
    c6_row: Exp464MethodFoldMetricAntigravity,
) -> Mapping[str, Mapping[str, int]]:
    diagnostics: dict[str, Mapping[str, int]] = {}
    for change_type in _PAIR_TYPES_ANTIGRAVITY:
        total_pairs = len(c0_row.normal_scores) * len(c0_row.primary_scores[change_type])
        fixed_pairs = 0
        broken_pairs = 0
        for normal_index, c0_normal_score in enumerate(c0_row.normal_scores):
            c6_normal_score = c6_row.normal_scores[normal_index]
            for change_index, c0_change_score in enumerate(c0_row.primary_scores[change_type]):
                c6_change_score = c6_row.primary_scores[change_type][change_index]
                if c0_change_score == c0_normal_score or c6_change_score == c6_normal_score:
                    continue
                correct_c0 = c0_change_score > c0_normal_score
                correct_c6 = c6_change_score > c6_normal_score
                if (not correct_c0) and correct_c6:
                    fixed_pairs += 1
                elif correct_c0 and (not correct_c6):
                    broken_pairs += 1
        diagnostics[change_type] = MappingProxyType(
            {
                "total_pairs": total_pairs,
                "fixed_pairs": fixed_pairs,
                "broken_pairs": broken_pairs,
            }
        )
    return MappingProxyType(diagnostics)


def _historical_period_threshold_antigravity(
    seed: int,
    fold_index: int,
) -> HistoricalPeriodThresholdAntigravity | None:
    for item in FROZEN_CONFIG_ANTIGRAVITY.historical_period_fold_thresholds:
        if item.seed == seed and item.fold_index == fold_index:
            return item
    return None


def _apply_c6_pair_and_comparison_gates_antigravity(
    rows: tuple[Exp464MethodFoldMetricAntigravity, ...],
) -> tuple[Exp464MethodFoldMetricAntigravity, ...]:
    c0_lookup = {
        (row.seed, row.fold_index): row
        for row in rows
        if row.method == "C0"
    }
    updated_rows: list[Exp464MethodFoldMetricAntigravity] = []
    for row in rows:
        if row.method != "C6":
            updated_rows.append(row)
            continue
        c0_row = c0_lookup[(row.seed, row.fold_index)]
        reasons = list(row.failure_reasons)
        if row.per_type_auc["gradual_drift"] < c0_row.per_type_auc["gradual_drift"]:
            reasons.append("gradual_drift_auc_below_c0")
        if row.per_type_auc["period_change"] < c0_row.per_type_auc["period_change"]:
            reasons.append("period_change_auc_below_c0")
        historical = _historical_period_threshold_antigravity(row.seed, row.fold_index)
        if historical is not None and row.per_type_auc["period_change"] < historical.period_change_auc_min:
            reasons.append(
                f"historical_period_change_below_min:{row.seed}:{row.fold_index}"
            )
        updated_rows.append(
            Exp464MethodFoldMetricAntigravity(
                method=row.method,
                seed=row.seed,
                fold_index=row.fold_index,
                normal_trials=row.normal_trials,
                normal_alerts=row.normal_alerts,
                normal_wilson_upper=row.normal_wilson_upper,
                per_type_auc=row.per_type_auc,
                mean_auc=row.mean_auc,
                scores_finite=row.scores_finite,
                normal_scores=row.normal_scores,
                primary_scores=row.primary_scores,
                confirmation_counts=row.confirmation_counts,
                confirmation_rates=row.confirmation_rates,
                pair_diagnostics=_pair_diagnostics_for_rows_antigravity(c0_row, row),
                passed=(not reasons),
                failure_reasons=tuple(sorted(set(reasons))),
                feature_dimension=row.feature_dimension,
                provenance=row.provenance,
                fit_amplitudes=row.fit_amplitudes,
                query_amplitudes=row.query_amplitudes,
                query_cosines=row.query_cosines,
                raw_evidence_kind=row.raw_evidence_kind,
                raw_evidence_source_method=row.raw_evidence_source_method,
                raw_fit_vector_blob=row.raw_fit_vector_blob,
                raw_query_vector_blobs=row.raw_query_vector_blobs,
            )
        )
    return tuple(updated_rows)


def summarize_methods_antigravity(
    rows: tuple[Exp464MethodFoldMetricAntigravity, ...],
) -> tuple[Exp464MethodSummaryAntigravity, ...]:
    if type(rows) is not tuple or len(rows) != 90:
        raise ValueError("summary input must contain exactly 90 ordered rows")
    summaries: list[Exp464MethodSummaryAntigravity] = []
    expected_rows = len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
    for method in _METHODS_ANTIGRAVITY:
        method_rows = tuple(row for row in rows if row.method == method)
        if len(method_rows) != expected_rows:
            raise ValueError("each method summary must cover ten seed-fold rows")
        minima = MappingProxyType(
            {
                change_type: min(row.per_type_auc[change_type] for row in method_rows)
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            }
        )
        summaries.append(
            Exp464MethodSummaryAntigravity(
                method=method,
                fold_count=len(method_rows),
                passed_count=sum(row.passed for row in method_rows),
                failed_count=sum(not row.passed for row in method_rows),
                min_per_type_auc=minima,
                min_gradual_drift_auc=minima["gradual_drift"],
                min_period_change_auc=minima["period_change"],
                min_mean_auc=min(row.mean_auc for row in method_rows),
                max_normal_wilson_upper=max(row.normal_wilson_upper for row in method_rows),
                all_scores_finite=all(row.scores_finite for row in method_rows),
                passed=all(row.passed for row in method_rows),
            )
        )
    return tuple(summaries)


def _exact_control_row_digest_antigravity(row: Exp464MethodFoldMetricAntigravity) -> str:
    payload = {
        "method": row.method,
        "seed": row.seed,
        "fold_index": row.fold_index,
        "normal_scores": list(row.normal_scores),
        "primary_scores": {
            change_type: list(row.primary_scores[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
    }
    return __import__("hashlib").sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()


def _validate_control_row_digests_antigravity(
    rows: tuple[Exp464MethodFoldMetricAntigravity, ...],
    authority: Exp463AuthorityAntigravity,
) -> None:
    for row in rows:
        if row.method not in ("C0", "C1", "C2", "C7"):
            continue
        key = f"{row.method}:{row.seed}:{row.fold_index}"
        if _exact_control_row_digest_antigravity(row) != authority.control_row_digests[key]:
            raise ValueError(f"{key} control row differs from authoritative Exp463 replay")


def _confirmed_scores_from_evidence_antigravity(
    baseline: tuple[float, ...],
    fit_amplitudes: tuple[float, ...],
    query_amplitudes: tuple[float, ...],
    query_cosines: tuple[float, ...],
) -> tuple[tuple[float, ...], int]:
    ranks = anomaly_rank_antigravity(
        np.asarray(fit_amplitudes, dtype=np.float64),
        np.asarray(query_amplitudes, dtype=np.float64),
    )
    confirmed = np.asarray(query_cosines, dtype=np.float64) >= FROZEN_CONFIG_ANTIGRAVITY.cosine_min
    scores = np.where(
        confirmed,
        np.maximum(np.asarray(baseline, dtype=np.float64), ranks),
        np.asarray(baseline, dtype=np.float64),
    )
    return tuple(float(value) for value in scores), int(np.count_nonzero(confirmed))


def _score_tuples_match_antigravity(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        abs(left - right) <= 1e-12
        for left, right in zip(actual, expected, strict=True)
    )


def _replayed_raw_evidence_antigravity(
    source_row: Exp464MethodFoldMetricAntigravity,
) -> tuple[tuple[float, ...], Mapping[str, tuple[float, ...]], Mapping[str, tuple[float, ...]]]:
    fit_matrix = decode_vector_matrix_antigravity(
        source_row.raw_fit_vector_blob,
        f"{source_row.method}:{source_row.seed}:{source_row.fold_index}:raw_fit_vector_blob",
    )
    fit_amplitudes, _fit_cosines = amplitudes_cosines_from_vectors_antigravity(
        source_row.raw_evidence_kind,
        fit_matrix,
    )
    query_amplitudes: dict[str, tuple[float, ...]] = {}
    query_cosines: dict[str, tuple[float, ...]] = {}
    for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
        query_matrix = decode_vector_matrix_antigravity(
            source_row.raw_query_vector_blobs[scope],
            (
                f"{source_row.method}:{source_row.seed}:{source_row.fold_index}:"
                f"raw_query_vector_blobs[{scope}]"
            ),
        )
        amplitudes, cosines = amplitudes_cosines_from_vectors_antigravity(
            source_row.raw_evidence_kind,
            query_matrix,
        )
        query_amplitudes[scope] = tuple(float(value) for value in amplitudes)
        query_cosines[scope] = tuple(float(value) for value in cosines)
    return (
        tuple(float(value) for value in fit_amplitudes),
        MappingProxyType(query_amplitudes),
        MappingProxyType(query_cosines),
    )


def _expected_counts_rates_and_scores_antigravity(
    row: Exp464MethodFoldMetricAntigravity,
    c0_row: Exp464MethodFoldMetricAntigravity | None,
    *,
    fit_amplitudes: tuple[float, ...],
    query_amplitudes: Mapping[str, tuple[float, ...]],
    query_cosines: Mapping[str, tuple[float, ...]],
) -> tuple[Mapping[str, int], Mapping[str, float], tuple[float, ...], Mapping[str, tuple[float, ...]]]:
    if row.method not in _EVIDENCE_METHODS_ANTIGRAVITY:
        zero_counts = MappingProxyType({name: 0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY})
        zero_rates = _confirmation_rates_from_counts_antigravity(zero_counts, row.normal_trials)
        return zero_counts, zero_rates, row.normal_scores, row.primary_scores
    assert c0_row is not None
    if row.method == "C3":
        expected_original = tuple(
            float(value)
            for value in anomaly_rank_antigravity(
                np.asarray(fit_amplitudes, dtype=np.float64),
                np.asarray(query_amplitudes["normal"], dtype=np.float64),
            )
        )
        expected_primary = MappingProxyType(
            {
                change_type: tuple(
                    float(value)
                    for value in anomaly_rank_antigravity(
                        np.asarray(fit_amplitudes, dtype=np.float64),
                        np.asarray(query_amplitudes[change_type], dtype=np.float64),
                    )
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            }
        )
        zero_counts = MappingProxyType({name: 0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY})
        return (
            zero_counts,
            _confirmation_rates_from_counts_antigravity(zero_counts, row.normal_trials),
            expected_original,
            expected_primary,
        )
    if row.method == "C4":
        expected_original = tuple(
            float(value)
            for value in np.maximum(
                np.asarray(c0_row.normal_scores, dtype=np.float64),
                anomaly_rank_antigravity(
                    np.asarray(fit_amplitudes, dtype=np.float64),
                    np.asarray(query_amplitudes["normal"], dtype=np.float64),
                ),
            )
        )
        expected_primary = MappingProxyType(
            {
                change_type: tuple(
                    float(value)
                    for value in np.maximum(
                        np.asarray(c0_row.primary_scores[change_type], dtype=np.float64),
                        anomaly_rank_antigravity(
                            np.asarray(fit_amplitudes, dtype=np.float64),
                            np.asarray(query_amplitudes[change_type], dtype=np.float64),
                        ),
                    )
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            }
        )
        zero_counts = MappingProxyType({name: 0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY})
        return (
            zero_counts,
            _confirmation_rates_from_counts_antigravity(zero_counts, row.normal_trials),
            expected_original,
            expected_primary,
        )
    expected_original, normal_count = _confirmed_scores_from_evidence_antigravity(
        c0_row.normal_scores,
        fit_amplitudes,
        query_amplitudes["normal"],
        query_cosines["normal"],
    )
    expected_primary_values: dict[str, tuple[float, ...]] = {}
    counts = {"normal": normal_count}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        expected_primary_values[change_type], counts[change_type] = _confirmed_scores_from_evidence_antigravity(
            c0_row.primary_scores[change_type],
            fit_amplitudes,
            query_amplitudes[change_type],
            query_cosines[change_type],
        )
    expected_counts = MappingProxyType(counts)
    return (
        expected_counts,
        _confirmation_rates_from_counts_antigravity(expected_counts, row.normal_trials),
        expected_original,
        MappingProxyType(expected_primary_values),
    )


def _apply_evidence_consistency_antigravity(
    rows: tuple[Exp464MethodFoldMetricAntigravity, ...],
) -> tuple[Exp464MethodFoldMetricAntigravity, ...]:
    c0_lookup = {
        (row.seed, row.fold_index): row for row in rows if row.method == "C0"
    }
    c6_lookup = {
        (row.seed, row.fold_index): row for row in rows if row.method == "C6"
    }
    updated: list[Exp464MethodFoldMetricAntigravity] = []
    for row in rows:
        replayed_fit_amplitudes = row.fit_amplitudes
        replayed_query_amplitudes = row.query_amplitudes
        replayed_query_cosines = row.query_cosines
        if row.method in _EVIDENCE_METHODS_ANTIGRAVITY:
            if row.method in _RAW_CARRIER_METHODS_ANTIGRAVITY:
                source_row = row
            else:
                source_row = c6_lookup.get((row.seed, row.fold_index))
                if source_row is None:
                    raise ValueError("C3/C4 raw evidence source C6 is missing")
            replayed_fit_amplitudes, replayed_query_amplitudes, replayed_query_cosines = _replayed_raw_evidence_antigravity(
                source_row
            )
            if not _score_tuples_match_antigravity(row.fit_amplitudes, replayed_fit_amplitudes):
                raise ValueError(
                    f"{row.method}:{row.seed}:{row.fold_index} fit amplitudes differ from raw replay"
                )
            for scope in _CONFIRMATION_TYPES_ANTIGRAVITY:
                if not _score_tuples_match_antigravity(
                    row.query_amplitudes[scope],
                    replayed_query_amplitudes[scope],
                ):
                    raise ValueError(
                        (
                            f"{row.method}:{row.seed}:{row.fold_index} "
                            f"query amplitudes[{scope}] differ from raw replay"
                        )
                    )
                if not _score_tuples_match_antigravity(
                    row.query_cosines[scope],
                    replayed_query_cosines[scope],
                ):
                    raise ValueError(
                        (
                            f"{row.method}:{row.seed}:{row.fold_index} "
                            f"query cosines[{scope}] differ from raw replay"
                        )
                    )
        expected_counts, expected_rates, expected_original, expected_primary = _expected_counts_rates_and_scores_antigravity(
            row,
            c0_lookup.get((row.seed, row.fold_index)),
            fit_amplitudes=replayed_fit_amplitudes,
            query_amplitudes=replayed_query_amplitudes,
            query_cosines=replayed_query_cosines,
        )
        if tuple(row.normal_scores) != expected_original:
            raise ValueError(f"{row.method}:{row.seed}:{row.fold_index} normal scores differ from evidence replay")
        if any(
            tuple(row.primary_scores[change_type]) != expected_primary[change_type]
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        ):
            raise ValueError(f"{row.method}:{row.seed}:{row.fold_index} primary scores differ from evidence replay")
        updated.append(
            Exp464MethodFoldMetricAntigravity(
                method=row.method,
                seed=row.seed,
                fold_index=row.fold_index,
                normal_trials=row.normal_trials,
                normal_alerts=row.normal_alerts,
                normal_wilson_upper=row.normal_wilson_upper,
                per_type_auc=row.per_type_auc,
                mean_auc=row.mean_auc,
                scores_finite=row.scores_finite,
                normal_scores=row.normal_scores,
                primary_scores=row.primary_scores,
                confirmation_counts=expected_counts,
                confirmation_rates=expected_rates,
                pair_diagnostics=row.pair_diagnostics,
                passed=row.passed,
                failure_reasons=row.failure_reasons,
                feature_dimension=row.feature_dimension,
                provenance=row.provenance,
                fit_amplitudes=replayed_fit_amplitudes,
                query_amplitudes=replayed_query_amplitudes,
                query_cosines=replayed_query_cosines,
                raw_evidence_kind=row.raw_evidence_kind,
                raw_evidence_source_method=row.raw_evidence_source_method,
                raw_fit_vector_blob=row.raw_fit_vector_blob,
                raw_query_vector_blobs=row.raw_query_vector_blobs,
            )
        )
    return tuple(updated)


def normalize_qualification_antigravity(
    qualification: Exp464QualificationContractAntigravity,
    authority: Exp463AuthorityAntigravity,
) -> Exp464QualificationContractAntigravity:
    if type(qualification) is not Exp464QualificationContractAntigravity:
        raise ValueError("qualification type is invalid")
    validate_exp463_authority_antigravity(authority)
    recomputed_rows = tuple(
        recompute_metric_from_scores_antigravity(row) for row in qualification.fold_metrics
    )
    _validate_control_row_digests_antigravity(recomputed_rows, authority)
    evidence_rows = _apply_evidence_consistency_antigravity(recomputed_rows)
    c6_rows = _apply_c6_pair_and_comparison_gates_antigravity(evidence_rows)
    return Exp464QualificationContractAntigravity(
        dataset=qualification.dataset,
        method_order=qualification.method_order,
        primary_candidate=qualification.primary_candidate,
        fold_metrics=c6_rows,
        summaries=summarize_methods_antigravity(c6_rows),
        label_query_count=qualification.label_query_count,
    )


def validate_control_reproduction_antigravity(
    qualification: Exp464QualificationContractAntigravity,
    authority: Exp463AuthorityAntigravity,
) -> tuple[str, ...]:
    validate_exp463_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(qualification, authority)
    _validate_control_row_digests_antigravity(normalized.fold_metrics, authority)
    return ()


def _metric_row_antigravity(
    scores: object,
    *,
    method: str,
    seed: int,
    fold_index: int,
) -> Exp464MethodFoldMetricAntigravity:
    if not all(
        hasattr(scores, attribute)
        for attribute in (
            "methods",
            "original",
            "primary",
            "confirmation_counts",
            "feature_dimensions",
            "fit_amplitudes",
            "query_amplitudes",
            "query_cosines",
            "raw_evidence_kind",
            "raw_evidence_source_method",
            "raw_fit_vector_blobs",
            "raw_query_vector_blobs",
            "provenance",
        )
    ):
        raise ValueError("fold_scorer returned an invalid score container")
    if getattr(scores, "methods") != _METHODS_ANTIGRAVITY:
        raise ValueError("fold_scorer method order differs from C0-C8")
    original = getattr(scores, "original")
    primary = getattr(scores, "primary")
    counts = getattr(scores, "confirmation_counts")
    fit_amplitudes = getattr(scores, "fit_amplitudes")
    query_amplitudes = getattr(scores, "query_amplitudes")
    query_cosines = getattr(scores, "query_cosines")
    raw_evidence_kind = getattr(scores, "raw_evidence_kind")
    raw_evidence_source_method = getattr(scores, "raw_evidence_source_method")
    raw_fit_vector_blobs = getattr(scores, "raw_fit_vector_blobs")
    raw_query_vector_blobs = getattr(scores, "raw_query_vector_blobs")
    dimensions = getattr(scores, "feature_dimensions")
    provenance = getattr(scores, "provenance")
    if not all(isinstance(value, Mapping) for value in (original, primary, counts, fit_amplitudes, query_amplitudes, query_cosines, raw_evidence_kind, raw_evidence_source_method, raw_fit_vector_blobs, raw_query_vector_blobs, dimensions, provenance)):
        raise ValueError("fold_scorer containers are invalid")
    normal_scores = _score_tuple_antigravity(original[method], f"original[{method}]")
    feature_dimension = dimensions[method]
    if type(feature_dimension) is not int or feature_dimension < 1:
        raise ValueError("feature dimension must be a positive integer")
    source = provenance[method]
    if type(source) is not str or not source:
        raise ValueError("provenance must be a non-empty string")
    row = Exp464MethodFoldMetricAntigravity(
        method=method,
        seed=seed,
        fold_index=fold_index,
        normal_trials=len(normal_scores),
        normal_alerts=0,
        normal_wilson_upper=1.0,
        per_type_auc={change_type: 0.0 for change_type in _CHANGE_TYPES_ANTIGRAVITY},
        mean_auc=0.0,
        scores_finite=False,
        normal_scores=normal_scores,
        primary_scores=primary[method],
        confirmation_counts=counts[method],
        confirmation_rates={
            name: 0.0 for name in _CONFIRMATION_TYPES_ANTIGRAVITY
        },
        pair_diagnostics=_zero_pair_diagnostics_antigravity(),
        passed=False,
        failure_reasons=("normalization_pending",),
        feature_dimension=feature_dimension,
        provenance=source,
        fit_amplitudes=fit_amplitudes[method],
        query_amplitudes=query_amplitudes[method],
        query_cosines=query_cosines[method],
        raw_evidence_kind=raw_evidence_kind[method],
        raw_evidence_source_method=raw_evidence_source_method[method],
        raw_fit_vector_blob=raw_fit_vector_blobs[method],
        raw_query_vector_blobs=raw_query_vector_blobs[method],
    )
    return recompute_metric_from_scores_antigravity(row)


def _train_antigravity(train: object) -> np.ndarray:
    expected = (
        FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows,
        FROZEN_CONFIG_ANTIGRAVITY.normal2_width,
    )
    if (
        type(train) is not np.ndarray
        or train.shape != expected
        or train.dtype != np.dtype(np.float32)
        or not np.isfinite(train).all()
    ):
        raise ValueError("train must be the exact finite float32 normal_2 TRAIN matrix")
    return train


def _qualify_methods_with_dependencies_antigravity(
    *,
    authority: Exp463AuthorityAntigravity,
    loader: Callable[[], np.ndarray],
    fold_scorer: Callable[..., object],
) -> Exp464QualificationContractAntigravity:
    validate_exp463_authority_antigravity(authority)
    train = _train_antigravity(loader())
    folds = blocked_plan_antigravity(len(train))
    rows_by_method: dict[str, list[Exp464MethodFoldMetricAntigravity]] = {
        method: [] for method in _METHODS_ANTIGRAVITY
    }
    for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        coverage = np.zeros(len(train), dtype=np.uint8)
        for fold in folds:
            parameters = inspect.signature(fold_scorer).parameters
            if "authority" in parameters:
                scores = fold_scorer(
                    train,
                    fold.fit_indices,
                    fold.query_indices,
                    seed=seed,
                    fold_index=fold.fold_index,
                    authority=authority,
                )
            else:
                scores = fold_scorer(
                    train,
                    fold.fit_indices,
                    fold.query_indices,
                    seed=seed,
                    fold_index=fold.fold_index,
                )
            for method in _METHODS_ANTIGRAVITY:
                rows_by_method[method].append(
                    _metric_row_antigravity(
                        scores,
                        method=method,
                        seed=seed,
                        fold_index=fold.fold_index,
                    )
                )
            np.add.at(coverage, fold.query_indices, 1)
        if not np.all(coverage == 1):
            raise ValueError("blocked folds must cover every TRAIN row exactly once per seed")
    provisional_rows = tuple(
        row for method in _METHODS_ANTIGRAVITY for row in rows_by_method[method]
    )
    provisional = Exp464QualificationContractAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=_METHODS_ANTIGRAVITY,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=provisional_rows,
        summaries=summarize_methods_antigravity(provisional_rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )
    return normalize_qualification_antigravity(provisional, authority)


def qualify_methods_antigravity() -> Exp464QualificationContractAntigravity:
    from .predecessor_antigravity import capture_exp463_authority_antigravity
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_normal2_train_antigravity

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp463_authority_antigravity(),
        loader=load_normal2_train_antigravity,
        fold_scorer=score_fold_methods_antigravity,
    )


MethodFoldMetricAntigravity = Exp464MethodFoldMetricAntigravity
MethodSummaryAntigravity = Exp464MethodSummaryAntigravity
MethodQualificationAntigravity = Exp464QualificationContractAntigravity


__all__ = (
    "Exp464MethodFoldMetricAntigravity",
    "Exp464MethodSummaryAntigravity",
    "Exp464QualificationContractAntigravity",
    "MethodFoldMetricAntigravity",
    "MethodSummaryAntigravity",
    "MethodQualificationAntigravity",
    "_qualify_methods_with_dependencies_antigravity",
    "normalize_qualification_antigravity",
    "qualify_methods_antigravity",
    "recompute_metric_from_scores_antigravity",
    "summarize_methods_antigravity",
    "validate_control_reproduction_antigravity",
)
