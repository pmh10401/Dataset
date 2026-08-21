"""Safe aggregate-only dashboard projection for Exp464 Task 4."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import (
    Exp464QualificationContractAntigravity,
    normalize_qualification_antigravity,
)


def _summary_payloads_antigravity(
    qualification: Exp464QualificationContractAntigravity,
) -> list[dict[str, object]]:
    return [
        {
            "method": summary.method,
            "fold_count": summary.fold_count,
            "passed_count": summary.passed_count,
            "failed_count": summary.failed_count,
            "min_level_shift_auc": summary.min_per_type_auc["level_shift"],
            "min_gradual_drift_auc": summary.min_gradual_drift_auc,
            "min_amplitude_scaling_auc": summary.min_per_type_auc["amplitude_scaling"],
            "min_period_change_auc": summary.min_period_change_auc,
            "min_mean_auc": summary.min_mean_auc,
            "max_normal_wilson_upper": summary.max_normal_wilson_upper,
            "all_scores_finite": summary.all_scores_finite,
            "passed": summary.passed,
        }
        for summary in qualification.summaries
    ]


def export_dashboard_antigravity(
    qualification: Exp464QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    from .predecessor_antigravity import capture_exp463_authority_antigravity

    authority = capture_exp463_authority_antigravity()
    normalized = normalize_qualification_antigravity(qualification, authority)
    recomputed_decision = evaluate_qualification_antigravity(normalized, authority)
    if decision != recomputed_decision:
        raise ValueError("dashboard decision differs from evidence-derived decision")
    primary_rows = tuple(
        row for row in normalized.fold_metrics if row.method == "C6"
    )
    return {
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "primary_passed_count": decision.primary_passed_count,
        "primary_required_count": decision.primary_required_count,
        "eligible_for_routing_research": decision.eligible_for_routing_research,
        "method_summaries": _summary_payloads_antigravity(normalized),
        "failed_fold_count": sum(not row.passed for row in normalized.fold_metrics),
        "historical_period_floors": [
            {
                "seed": item.seed,
                "fold_index": item.fold_index,
                "required_period_change_auc_min": item.period_change_auc_min,
                "candidate_period_change_auc": next(
                    row.per_type_auc["period_change"]
                    for row in primary_rows
                    if row.seed == item.seed and row.fold_index == item.fold_index
                ),
            }
            for item in FROZEN_CONFIG_ANTIGRAVITY.historical_period_fold_thresholds
        ],
        "non_degradation_counts": {
            "gradual_rows_not_below_c0": sum(
                row.per_type_auc["gradual_drift"]
                >= next(
                    candidate.per_type_auc["gradual_drift"]
                    for candidate in normalized.fold_metrics
                    if candidate.method == "C0"
                    and candidate.seed == row.seed
                    and candidate.fold_index == row.fold_index
                )
                for row in primary_rows
            ),
            "period_rows_not_below_c0": sum(
                row.per_type_auc["period_change"]
                >= next(
                    candidate.per_type_auc["period_change"]
                    for candidate in normalized.fold_metrics
                    if candidate.method == "C0"
                    and candidate.seed == row.seed
                    and candidate.fold_index == row.fold_index
                )
                for row in primary_rows
            ),
        },
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    }


__all__ = ("export_dashboard_antigravity",)
