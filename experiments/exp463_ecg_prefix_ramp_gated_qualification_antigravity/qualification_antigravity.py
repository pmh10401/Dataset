"""TRAIN-only blocked qualification and evidence normalization for Exp463."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    Exp462AuthorityAntigravity,
    Exp463MethodFoldMetricAntigravity,
    Exp463MethodSummaryAntigravity,
    Exp463QualificationContractAntigravity,
    validate_exp462_authority_antigravity,
)

if TYPE_CHECKING:
    from .scoring_antigravity import FoldMethodScoresAntigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_order
_CONTROL_METHODS_ANTIGRAVITY = {"M0": 0, "M6": 2}


def _empirical_pvalues_antigravity(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return the frozen upper-tail plus-one p-value for each score."""

    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    counts = len(ordered) - np.searchsorted(ordered, query, side="left")
    return np.asarray((1.0 + counts) / (1.0 + len(ordered)), dtype=np.float64)


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
    return tuple(float(item) for item in array)


def _primary_scores_antigravity(
    value: Mapping[str, object],
    expected_len: int,
) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError("primary_scores must preserve exact change-type order")
    result: dict[str, tuple[float, ...]] = {}
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        scores = _score_tuple_antigravity(value[change_type], f"primary_scores[{change_type}]")
        if len(scores) != expected_len:
            raise ValueError("primary score lengths must equal normal score length")
        result[change_type] = scores
    return MappingProxyType(result)


def _safe_metrics_antigravity(
    normal: np.ndarray,
    primary: Mapping[str, np.ndarray],
) -> tuple[int, float, dict[str, float], float, bool]:
    """Compute every metric from evidence, retaining a finite failure sentinel."""

    finite = bool(
        np.isfinite(normal).all()
        and all(np.isfinite(primary[change_type]).all() for change_type in _CHANGE_TYPES_ANTIGRAVITY)
    )
    if not finite:
        return (
            0,
            1.0,
            {change_type: 0.0 for change_type in _CHANGE_TYPES_ANTIGRAVITY},
            0.0,
            False,
        )
    alerts = int(
        np.count_nonzero(
            _empirical_pvalues_antigravity(normal, normal)
            <= FROZEN_CONFIG_ANTIGRAVITY.alpha
        )
    )
    per_type = {
        change_type: float(mann_whitney_auc_antigravity(normal, primary[change_type]))
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    pooled = float(sum(per_type.values()) / len(_CHANGE_TYPES_ANTIGRAVITY))
    return alerts, float(wilson_upper_antigravity(alerts, len(normal))), per_type, pooled, True


def _metric_from_values_antigravity(
    row: Exp463MethodFoldMetricAntigravity,
    *,
    normal: np.ndarray,
    primary: Mapping[str, np.ndarray],
    authority: Exp462AuthorityAntigravity,
) -> Exp463MethodFoldMetricAntigravity:
    alerts, wilson, per_type, pooled, finite = _safe_metrics_antigravity(normal, primary)
    candidate = Exp463MethodFoldMetricAntigravity(
        method=row.method,
        seed=row.seed,
        fold_index=row.fold_index,
        normal_trials=len(normal),
        normal_alerts=alerts,
        normal_wilson_upper=float(wilson),
        per_type_auc=per_type,
        pooled_auc=float(pooled),
        scores_finite=finite,
        normal_scores=tuple(float(item) for item in normal),
        primary_scores={
            change_type: tuple(float(item) for item in primary[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        passed=True,
        failure_reasons=(),
    )
    passed, reasons = recompute_row_pass_antigravity(candidate, authority)
    return Exp463MethodFoldMetricAntigravity(
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


def recompute_row_pass_antigravity(
    row: Exp463MethodFoldMetricAntigravity,
    authority: Exp462AuthorityAntigravity,
) -> tuple[bool, tuple[str, ...]]:
    """Apply the frozen row gates without trusting stored pass metadata."""

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
    if row.method in {"M2", "M6"}:
        if row.per_type_auc["gradual_drift"] <= FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_strictly_above_exp462:
            reasons.append(
                "gradual_drift_not_strictly_improved_over_exp462"
                if row.method == "M2"
                else "m6_gradual_drift_not_strictly_improved_over_exp462"
            )
        if row.per_type_auc["period_change"] < FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min:
            reasons.append(
                "period_change_auc_below_global_min"
                if row.method == "M2"
                else "m6_period_change_auc_below_global_min"
            )
        for historical in authority.historical_period_folds:
            if row.seed == historical.seed and row.fold_index == historical.fold_index:
                if row.per_type_auc["period_change"] < historical.period_change_auc:
                    reasons.append(
                        (
                            f"historical_period_change_below_min:{row.seed}:{row.fold_index}"
                            if row.method == "M2"
                            else f"m6_historical_period_change_below_min:{row.seed}:{row.fold_index}"
                        )
                    )
                break
    return not reasons, tuple(reasons)


def recompute_metric_from_scores_antigravity(
    row: Exp463MethodFoldMetricAntigravity,
    authority: Exp462AuthorityAntigravity,
) -> Exp463MethodFoldMetricAntigravity:
    """Rebuild metrics, summaries, and pass reasons solely from score evidence."""

    trusted_authority = validate_exp462_authority_antigravity(authority)
    normal = np.asarray(row.normal_scores, dtype=np.float64)
    primary = {
        change_type: np.asarray(row.primary_scores[change_type], dtype=np.float64)
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    return _metric_from_values_antigravity(
        row,
        normal=normal,
        primary=primary,
        authority=trusted_authority,
    )


def summarize_methods_antigravity(
    rows: tuple[Exp463MethodFoldMetricAntigravity, ...],
) -> tuple[Exp463MethodSummaryAntigravity, ...]:
    """Return exactly seven summaries in frozen M0–M6 order."""

    if type(rows) is not tuple or len(rows) != 70:
        raise ValueError("summary input must contain exactly 70 ordered rows")
    summaries: list[Exp463MethodSummaryAntigravity] = []
    expected_rows = len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
    for method in _METHODS_ANTIGRAVITY:
        method_rows = tuple(row for row in rows if row.method == method)
        if len(method_rows) != expected_rows:
            raise ValueError("each method summary must cover ten seed-fold rows")
        minima = {
            change_type: min(row.per_type_auc[change_type] for row in method_rows)
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
        summaries.append(
            Exp463MethodSummaryAntigravity(
                method=method,
                fold_count=len(method_rows),
                passed_count=sum(row.passed for row in method_rows),
                failed_count=sum(not row.passed for row in method_rows),
                min_per_type_auc=minima,
                min_gradual_drift_auc=minima["gradual_drift"],
                min_period_change_auc=minima["period_change"],
                min_pooled_auc=min(row.pooled_auc for row in method_rows),
                max_normal_wilson_upper=max(row.normal_wilson_upper for row in method_rows),
                all_scores_finite=all(row.scores_finite for row in method_rows),
                passed=all(row.passed for row in method_rows),
            )
        )
    return tuple(summaries)


def normalize_qualification_antigravity(
    authority: Exp462AuthorityAntigravity,
    qualification: Exp463QualificationContractAntigravity,
) -> Exp463QualificationContractAntigravity:
    """Normalize caller-provided rows and recompute all seven summaries."""

    if type(qualification) is not Exp463QualificationContractAntigravity:
        raise ValueError("qualification type is invalid")
    trusted_authority = validate_exp462_authority_antigravity(authority)
    rows = tuple(
        recompute_metric_from_scores_antigravity(row, trusted_authority)
        for row in qualification.fold_metrics
    )
    return Exp463QualificationContractAntigravity(
        dataset=qualification.dataset,
        method_order=qualification.method_order,
        primary_candidate=qualification.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=qualification.label_query_count,
    )


def _summary_matches_control_antigravity(
    summary: Exp463MethodSummaryAntigravity,
    expected: object,
    *,
    expected_primary_minima: Mapping[str, float] | None = None,
) -> bool:
    expected_method = getattr(expected, "method")
    matches = (
        summary.method == ("M0" if expected_method == "hydra_period_rank_max" else "M6")
        and summary.fold_count == getattr(expected, "fold_count")
        and summary.passed_count == getattr(expected, "passed_count")
        and summary.failed_count == getattr(expected, "failed_count")
        and summary.min_gradual_drift_auc == getattr(expected, "min_gradual_drift_auc")
        and summary.min_period_change_auc == getattr(expected, "min_period_change_auc")
        and summary.min_pooled_auc == getattr(expected, "min_pooled_auc")
        and summary.max_normal_wilson_upper == getattr(expected, "max_normal_wilson_upper")
        and summary.all_scores_finite == getattr(expected, "all_scores_finite")
        and summary.passed == getattr(expected, "passed")
    )
    if expected_primary_minima is not None:
        matches = matches and dict(summary.min_per_type_auc) == dict(expected_primary_minima)
    return matches


def validate_control_reproduction_antigravity(
    qualification: Exp463QualificationContractAntigravity,
    authority: Exp462AuthorityAntigravity,
) -> tuple[str, ...]:
    """Require M0 and M6 aggregate controls to equal frozen predecessor summaries."""

    trusted_authority = validate_exp462_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    expected_summaries = trusted_authority.result_snapshot.summaries
    actual_by_method = {summary.method: summary for summary in normalized.summaries}
    reasons: list[str] = []
    for method, expected_index in _CONTROL_METHODS_ANTIGRAVITY.items():
        expected = expected_summaries[expected_index]
        expected_primary_minima = (
            trusted_authority.result_snapshot.primary_min_per_type_auc
            if method == "M6"
            else None
        )
        if not _summary_matches_control_antigravity(
            actual_by_method[method],
            expected,
            expected_primary_minima=expected_primary_minima,
        ):
            reasons.append(f"{method.lower()}_control_not_reproduced")
    return tuple(reasons)


def _metric_row_antigravity(
    authority: Exp462AuthorityAntigravity,
    scores: FoldMethodScoresAntigravity,
    *,
    method: str,
    seed: int,
    fold_index: int,
) -> Exp463MethodFoldMetricAntigravity:
    normal = np.asarray(scores.original[method], dtype=np.float64)
    primary = {
        change_type: np.asarray(scores.primary[method][change_type], dtype=np.float64)
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    row = Exp463MethodFoldMetricAntigravity(
        method=method,
        seed=seed,
        fold_index=fold_index,
        normal_trials=len(normal),
        normal_alerts=0,
        normal_wilson_upper=1.0,
        per_type_auc={change_type: 0.0 for change_type in _CHANGE_TYPES_ANTIGRAVITY},
        pooled_auc=0.0,
        scores_finite=False,
        normal_scores=tuple(float(item) for item in normal),
        primary_scores={
            change_type: tuple(float(item) for item in primary[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
        passed=False,
        failure_reasons=("normalization_pending",),
    )
    return _metric_from_values_antigravity(
        row,
        normal=normal,
        primary=primary,
        authority=authority,
    )


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
    authority: Exp462AuthorityAntigravity,
    loader: Callable[[], np.ndarray],
    fold_scorer: Callable[..., FoldMethodScoresAntigravity],
) -> Exp463QualificationContractAntigravity:
    """Build ordered score-derived rows from injected source/scorer boundaries."""

    from .scoring_antigravity import FoldMethodScoresAntigravity

    trusted_authority = validate_exp462_authority_antigravity(authority)
    train = _train_antigravity(loader())
    folds = blocked_plan_antigravity(len(train))
    rows_by_method: dict[str, list[Exp463MethodFoldMetricAntigravity]] = {
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
            if scores.methods != _METHODS_ANTIGRAVITY:
                raise ValueError("fold_scorer method order differs from M0-M6")
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
    return Exp463QualificationContractAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        method_order=_METHODS_ANTIGRAVITY,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        fold_metrics=rows,
        summaries=summarize_methods_antigravity(rows),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )


def qualify_methods_antigravity() -> Exp463QualificationContractAntigravity:
    """Run the frozen TRAIN-only qualification boundary when explicitly invoked."""

    from .predecessor_antigravity import capture_exp462_authority_antigravity
    from .scoring_antigravity import score_fold_methods_antigravity
    from .sources_antigravity import load_normal2_train_antigravity

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp462_authority_antigravity(),
        loader=load_normal2_train_antigravity,
        fold_scorer=score_fold_methods_antigravity,
    )


# Descriptive names mirror the Task 2 immutable contracts without introducing
# another mutable container implementation.
MethodFoldMetricAntigravity = Exp463MethodFoldMetricAntigravity
MethodSummaryAntigravity = Exp463MethodSummaryAntigravity
MethodQualificationAntigravity = Exp463QualificationContractAntigravity


__all__ = (
    "Exp463MethodFoldMetricAntigravity",
    "Exp463MethodSummaryAntigravity",
    "Exp463QualificationContractAntigravity",
    "MethodFoldMetricAntigravity",
    "MethodSummaryAntigravity",
    "MethodQualificationAntigravity",
    "normalize_qualification_antigravity",
    "qualify_methods_antigravity",
    "recompute_metric_from_scores_antigravity",
    "recompute_row_pass_antigravity",
    "summarize_methods_antigravity",
    "validate_control_reproduction_antigravity",
    "_qualify_methods_with_dependencies_antigravity",
)
