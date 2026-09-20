"""TRAIN-only blocked qualification for the Exp462 gradual signature methods."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
    blocked_plan_antigravity,
)
from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
    mann_whitney_auc_antigravity,
    wilson_upper_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp461GradualAuthorityAntigravity,
    validate_exp461_gradual_authority_antigravity,
)
from .predecessor_antigravity import capture_exp461_gradual_authority_antigravity

if TYPE_CHECKING:
    from .scoring_antigravity import FoldMethodScoresAntigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_order


def _empirical_pvalues_antigravity(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Frozen upper-tail plus-one p-values without importing the heavy scorer stack."""

    ordered = np.sort(reference)
    counts = len(ordered) - np.searchsorted(ordered, query, side="left")
    return np.asarray((1.0 + counts) / (1.0 + len(ordered)), dtype=np.float64)


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _int_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed_antigravity(value: object) -> int:
    if type(value) is not int or value not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the frozen seeds")
    return value


def _fold_index_antigravity(value: object) -> int:
    if type(value) is not int or not 0 <= value < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be one of the frozen folds")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _per_type_auc_antigravity(value: Mapping[str, object]) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_CHANGE_TYPES_ANTIGRAVITY):
        raise ValueError("per_type_auc must contain the exact change types")
    return MappingProxyType(
        {
            change_type: _unit_float_antigravity(
                value[change_type],
                f"per_type_auc[{change_type}]",
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )


def _score_tuple_antigravity(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64)
    elif type(value) in (tuple, list):
        try:
            array = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a score vector") from error
    else:
        raise ValueError(f"{name} must be a score vector")
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional score vector")
    # Non-finite evidence is retained so the finite-status gate can fail
    # semantically during normalization; canonical publication rejects it.
    return tuple(float(item) for item in array)


def _primary_scores_antigravity(
    value: Mapping[str, object],
    expected_len: int,
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError("primary_scores must contain the exact ordered change types")
    frozen: dict[str, tuple[float, ...]] = {}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        scores = _score_tuple_antigravity(value[change_type], f"primary_scores[{change_type}]")
        if len(scores) != expected_len:
            raise ValueError("primary score lengths must match normal scores")
        frozen[change_type] = scores
    return MappingProxyType(frozen)


def _method_antigravity(value: object) -> str:
    if type(value) is not str or value not in _METHODS_ANTIGRAVITY:
        raise ValueError("method must be one of the frozen methods")
    return value


def _failure_reasons_antigravity(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise ValueError("failure_reasons must be a tuple of non-empty strings")
    return value


@dataclass(frozen=True, slots=True)
class MethodFoldMetricAntigravity:
    method: str
    seed: int
    fold_index: int
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    per_type_auc: Mapping[str, float]
    pooled_auc: float
    scores_finite: bool
    normal_scores: tuple[float, ...]
    primary_scores: Mapping[str, tuple[float, ...]]
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _method_antigravity(self.method)
        _seed_antigravity(self.seed)
        _fold_index_antigravity(self.fold_index)
        trials = _int_antigravity(self.normal_trials, "normal_trials")
        if trials < 1:
            raise ValueError("normal_trials must be positive")
        alerts = _int_antigravity(self.normal_alerts, "normal_alerts")
        if alerts > trials:
            raise ValueError("normal_alerts must not exceed normal_trials")
        _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        per_type_auc = _per_type_auc_antigravity(self.per_type_auc)
        pooled_auc = _unit_float_antigravity(self.pooled_auc, "pooled_auc")
        expected_pooled = float(sum(per_type_auc.values()) / len(_CHANGE_TYPES_ANTIGRAVITY))
        if pooled_auc != expected_pooled:
            raise ValueError("pooled_auc must equal the mean per-type AUC")
        _bool_antigravity(self.scores_finite, "scores_finite")
        normal_scores = _score_tuple_antigravity(self.normal_scores, "normal_scores")
        if trials != len(normal_scores):
            raise ValueError("normal_trials must equal the normal score evidence length")
        primary_scores = _primary_scores_antigravity(self.primary_scores, len(normal_scores))
        _bool_antigravity(self.passed, "passed")
        _failure_reasons_antigravity(self.failure_reasons)
        object.__setattr__(self, "per_type_auc", per_type_auc)
        object.__setattr__(self, "normal_scores", normal_scores)
        object.__setattr__(self, "primary_scores", primary_scores)


@dataclass(frozen=True, slots=True)
class MethodSummaryAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_period_change_auc: float
    min_pooled_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool
    passed: bool

    def __post_init__(self) -> None:
        _method_antigravity(self.method)
        fold_count = _int_antigravity(self.fold_count, "fold_count")
        passed_count = _int_antigravity(self.passed_count, "passed_count")
        failed_count = _int_antigravity(self.failed_count, "failed_count")
        if fold_count != passed_count + failed_count:
            raise ValueError("summary counts must cover all folds")
        if fold_count < 1:
            raise ValueError("summary must contain at least one fold")
        _unit_float_antigravity(self.min_period_change_auc, "min_period_change_auc")
        _unit_float_antigravity(self.min_pooled_auc, "min_pooled_auc")
        _unit_float_antigravity(self.max_normal_wilson_upper, "max_normal_wilson_upper")
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class MethodQualificationAntigravity:
    dataset: str
    method_order: tuple[str, ...]
    primary_candidate: str
    fold_metrics: tuple[MethodFoldMetricAntigravity, ...]
    summaries: tuple[MethodSummaryAntigravity, ...]
    label_query_count: int

    def __post_init__(self) -> None:
        if self.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
            raise ValueError("dataset must be the frozen dataset")
        if self.method_order != _METHODS_ANTIGRAVITY:
            raise ValueError("method_order must preserve frozen order")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary candidate differs")
        expected_order = tuple(
            (method, seed, fold_index)
            for method in _METHODS_ANTIGRAVITY
            for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if type(self.fold_metrics) is not tuple:
            raise ValueError("fold_metrics must be a tuple")
        if any(type(row) is not MethodFoldMetricAntigravity for row in self.fold_metrics):
            raise ValueError("fold_metrics must contain MethodFoldMetricAntigravity")
        if tuple((row.method, row.seed, row.fold_index) for row in self.fold_metrics) != expected_order:
            raise ValueError("fold metric order differs from frozen method/seed/fold order")
        if type(self.summaries) is not tuple or tuple(summary.method for summary in self.summaries) != _METHODS_ANTIGRAVITY:
            raise ValueError("summaries must preserve method order")
        if any(type(summary) is not MethodSummaryAntigravity for summary in self.summaries):
            raise ValueError("summaries must contain MethodSummaryAntigravity")
        if _int_antigravity(self.label_query_count, "label_query_count") != 0:
            raise ValueError("label query count must remain zero")


def recompute_row_pass_antigravity(
    row: MethodFoldMetricAntigravity,
    authority: Exp461GradualAuthorityAntigravity,
) -> tuple[bool, tuple[str, ...]]:
    """Recompute every frozen row gate from the evidence-derived metrics."""

    reasons: list[str] = []
    if not row.scores_finite:
        reasons.append("nonfinite_scores")
    if row.normal_wilson_upper > FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max:
        reasons.append("normal_wilson_upper_above_max")
    if row.pooled_auc < FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min:
        reasons.append("pooled_auc_below_min")
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        if row.per_type_auc[change_type] < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"{change_type}_auc_below_min")
    if row.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
        if row.per_type_auc["gradual_drift"] <= FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc:
            # The common per-type gate is the hard floor; this reason records
            # the strict improvement requirement over Exp461 independently.
            reasons.append("gradual_drift_not_strictly_improved_over_exp461")
        if row.per_type_auc["period_change"] < FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min:
            reasons.append("period_change_auc_below_global_min")
        for historical in authority.historical_period_folds:
            if row.seed == historical.seed and row.fold_index == historical.fold_index:
                if row.per_type_auc["period_change"] < historical.period_change_auc:
                    reasons.append(
                        f"historical_period_change_below_min:{row.seed}:{row.fold_index}"
                    )
                break
    return not reasons, tuple(reasons)


def _safe_metrics_antigravity(
    normal: np.ndarray,
    primary: Mapping[str, np.ndarray],
) -> tuple[int, float, dict[str, float], float, bool]:
    scores_finite = bool(
        np.isfinite(normal).all()
        and all(np.isfinite(primary[change_type]).all() for change_type in _CHANGE_TYPES_ANTIGRAVITY)
    )
    if not scores_finite:
        return 0, 1.0, {change_type: 0.0 for change_type in _CHANGE_TYPES_ANTIGRAVITY}, 0.0, False
    normal_alerts = int(
        np.count_nonzero(
            _empirical_pvalues_antigravity(normal, normal) <= FROZEN_CONFIG_ANTIGRAVITY.alpha
        )
    )
    per_type_auc = {
        change_type: float(mann_whitney_auc_antigravity(normal, primary[change_type]))
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    return (
        normal_alerts,
        wilson_upper_antigravity(normal_alerts, len(normal)),
        per_type_auc,
        float(sum(per_type_auc.values()) / len(_CHANGE_TYPES_ANTIGRAVITY)),
        True,
    )


def recompute_metric_from_scores_antigravity(
    row: MethodFoldMetricAntigravity,
    authority: Exp461GradualAuthorityAntigravity,
) -> MethodFoldMetricAntigravity:
    """Rebuild metrics/pass reasons solely from stored score evidence."""

    trusted_authority = validate_exp461_gradual_authority_antigravity(authority)
    original = np.asarray(row.normal_scores, dtype=np.float64)
    primary = {
        change_type: np.asarray(row.primary_scores[change_type], dtype=np.float64)
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    normal_alerts, wilson, per_type_auc, pooled_auc, scores_finite = _safe_metrics_antigravity(
        original,
        primary,
    )
    candidate = MethodFoldMetricAntigravity(
        method=row.method,
        seed=row.seed,
        fold_index=row.fold_index,
        normal_trials=len(original),
        normal_alerts=normal_alerts,
        normal_wilson_upper=float(wilson),
        per_type_auc=per_type_auc,
        pooled_auc=float(pooled_auc),
        scores_finite=scores_finite,
        normal_scores=row.normal_scores,
        primary_scores=row.primary_scores,
        passed=True,
        failure_reasons=(),
    )
    passed, reasons = recompute_row_pass_antigravity(candidate, trusted_authority)
    return MethodFoldMetricAntigravity(
        method=candidate.method,
        seed=candidate.seed,
        fold_index=candidate.fold_index,
        normal_trials=candidate.normal_trials,
        normal_alerts=candidate.normal_alerts,
        normal_wilson_upper=candidate.normal_wilson_upper,
        per_type_auc=candidate.per_type_auc,
        pooled_auc=candidate.pooled_auc,
        scores_finite=candidate.scores_finite,
        normal_scores=candidate.normal_scores,
        primary_scores=candidate.primary_scores,
        passed=passed,
        failure_reasons=reasons,
    )


def summarize_methods_antigravity(
    rows: tuple[MethodFoldMetricAntigravity, ...],
) -> tuple[MethodSummaryAntigravity, ...]:
    summaries: list[MethodSummaryAntigravity] = []
    for method in _METHODS_ANTIGRAVITY:
        method_rows = tuple(row for row in rows if row.method == method)
        if len(method_rows) != len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("each method summary must cover all ten seed-fold rows")
        summaries.append(
            MethodSummaryAntigravity(
                method=method,
                fold_count=len(method_rows),
                passed_count=sum(row.passed for row in method_rows),
                failed_count=sum(not row.passed for row in method_rows),
                min_period_change_auc=min(row.per_type_auc["period_change"] for row in method_rows),
                min_pooled_auc=min(row.pooled_auc for row in method_rows),
                max_normal_wilson_upper=max(row.normal_wilson_upper for row in method_rows),
                all_scores_finite=all(row.scores_finite for row in method_rows),
                passed=all(row.passed for row in method_rows),
            )
        )
    return tuple(summaries)


def normalize_qualification_antigravity(
    authority: Exp461GradualAuthorityAntigravity,
    qualification: MethodQualificationAntigravity,
) -> MethodQualificationAntigravity:
    if type(qualification) is not MethodQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    trusted_authority = validate_exp461_gradual_authority_antigravity(authority)
    rows = tuple(
        recompute_metric_from_scores_antigravity(row, trusted_authority)
        for row in qualification.fold_metrics
    )
    return MethodQualificationAntigravity(
        dataset=qualification.dataset,
        method_order=qualification.method_order,
        primary_candidate=qualification.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=qualification.label_query_count,
    )


def _train_antigravity(train: object) -> np.ndarray:
    if (
        type(train) is not np.ndarray
        or train.shape
        != (
            FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows,
            FROZEN_CONFIG_ANTIGRAVITY.normal2_width,
        )
        or train.dtype != np.dtype(np.float32)
        or not np.isfinite(train).all()
    ):
        raise ValueError("train must be the exact finite float32 normal_2 TRAIN matrix")
    return train


def _metric_row_antigravity(
    authority: Exp461GradualAuthorityAntigravity,
    scores: FoldMethodScoresAntigravity,
    *,
    method: str,
    seed: int,
    fold_index: int,
) -> MethodFoldMetricAntigravity:
    original = np.asarray(scores.original[method], dtype=np.float64)
    primary = {
        change_type: np.asarray(scores.primary[method][change_type], dtype=np.float64)
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    normal_alerts, wilson, per_type_auc, pooled_auc, scores_finite = _safe_metrics_antigravity(
        original,
        primary,
    )
    candidate = MethodFoldMetricAntigravity(
        method=method,
        seed=seed,
        fold_index=fold_index,
        normal_trials=len(original),
        normal_alerts=normal_alerts,
        normal_wilson_upper=float(wilson),
        per_type_auc=per_type_auc,
        pooled_auc=float(pooled_auc),
        scores_finite=scores_finite,
        normal_scores=tuple(float(item) for item in original),
        primary_scores={
            change_type: tuple(float(item) for item in primary[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        passed=True,
        failure_reasons=(),
    )
    passed, reasons = recompute_row_pass_antigravity(candidate, authority)
    return MethodFoldMetricAntigravity(
        method=candidate.method,
        seed=candidate.seed,
        fold_index=candidate.fold_index,
        normal_trials=candidate.normal_trials,
        normal_alerts=candidate.normal_alerts,
        normal_wilson_upper=candidate.normal_wilson_upper,
        per_type_auc=candidate.per_type_auc,
        pooled_auc=candidate.pooled_auc,
        scores_finite=candidate.scores_finite,
        normal_scores=candidate.normal_scores,
        primary_scores=candidate.primary_scores,
        passed=passed,
        failure_reasons=reasons,
    )


def _qualify_methods_with_dependencies_antigravity(
    *,
    authority: Exp461GradualAuthorityAntigravity,
    loader: Callable[[], np.ndarray],
    fold_scorer: Callable[..., FoldMethodScoresAntigravity],
) -> MethodQualificationAntigravity:
    from .scoring_antigravity import FoldMethodScoresAntigravity

    trusted_authority = validate_exp461_gradual_authority_antigravity(authority)
    train = _train_antigravity(loader())
    folds = blocked_plan_antigravity(len(train))
    rows_by_method: dict[str, list[MethodFoldMetricAntigravity]] = {
        method: [] for method in _METHODS_ANTIGRAVITY
    }
    for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        coverage = np.zeros(len(train), dtype=np.uint8)
        for fold in folds:
            scores = fold_scorer(
                train,
                fold.fit_indices,
                fold.query_indices,
                seed=seed,
                fold_index=fold.fold_index,
            )
            if type(scores) is not FoldMethodScoresAntigravity:
                raise ValueError("fold_scorer returned an invalid score container")
            for method in _METHODS_ANTIGRAVITY:
                rows_by_method[method].append(
                    _metric_row_antigravity(
                        trusted_authority,
                        scores,
                        method=method,
                        seed=seed,
                        fold_index=fold.fold_index,
                    )
                )
            np.add.at(coverage, fold.query_indices, 1)
        if not np.all(coverage == 1):
            raise ValueError("blocked folds must cover every TRAIN row exactly once per seed")
    rows = tuple(row for method in _METHODS_ANTIGRAVITY for row in rows_by_method[method])
    return MethodQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=_METHODS_ANTIGRAVITY,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )


def qualify_methods_antigravity() -> MethodQualificationAntigravity:
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_normal2_train_antigravity

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp461_gradual_authority_antigravity(),
        loader=load_normal2_train_antigravity,
        fold_scorer=score_fold_methods_antigravity,
    )


__all__ = (
    "MethodFoldMetricAntigravity",
    "MethodQualificationAntigravity",
    "MethodSummaryAntigravity",
    "normalize_qualification_antigravity",
    "qualify_methods_antigravity",
    "recompute_metric_from_scores_antigravity",
    "recompute_row_pass_antigravity",
    "summarize_methods_antigravity",
)
