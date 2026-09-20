"""Restricted Exp461 Task 4 dashboard payload."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp460PeriodAuthorityAntigravity,
    validate_exp460_period_authority_antigravity,
)
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import (
    MethodQualificationAntigravity,
    normalize_qualification_antigravity,
)


def _summary_payloads_antigravity(
    authority: Exp460PeriodAuthorityAntigravity,
    qualification: MethodQualificationAntigravity,
) -> list[dict[str, object]]:
    normalized = normalize_qualification_antigravity(authority, qualification)
    return [
        {
            "method": summary.method,
            "fold_count": summary.fold_count,
            "passed_count": summary.passed_count,
            "failed_count": summary.failed_count,
            "min_period_change_auc": summary.min_period_change_auc,
            "min_pooled_auc": summary.min_pooled_auc,
            "max_normal_wilson_upper": summary.max_normal_wilson_upper,
            "all_scores_finite": summary.all_scores_finite,
            "passed": summary.passed,
        }
        for summary in normalized.summaries
    ]


def export_dashboard_antigravity(
    authority: Exp460PeriodAuthorityAntigravity,
    qualification: MethodQualificationAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    trusted_authority = validate_exp460_period_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    recomputed_decision = evaluate_qualification_antigravity(trusted_authority, normalized)
    if decision != recomputed_decision:
        raise ValueError("dashboard decision differs from recomputed qualification")
    method_summaries = _summary_payloads_antigravity(trusted_authority, normalized)
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "stage": "Task4 implemented pending external review / Task5 next",
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "primary_status": decision.status,
        "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
        "method_summaries": method_summaries,
        "failed_fold_count": sum(summary["failed_count"] for summary in method_summaries),
        "historical_improvements": [
            {
                "seed": failed.seed,
                "fold_index": failed.fold_index,
                "previous_period_change_auc": failed.period_change_auc,
                "candidate_period_change_auc": next(
                    row.per_type_auc["period_change"]
                    for row in normalized.fold_metrics
                    if row.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate
                    and row.seed == failed.seed
                    and row.fold_index == failed.fold_index
                ),
            }
            for failed in trusted_authority.failed_folds
        ],
        "decision": {
            "eligible_for_routing_research": decision.eligible_for_routing_research,
            "status": decision.status,
            "failure_reasons": list(decision.failure_reasons),
            "routing_applied": decision.routing_applied,
        },
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        "official_command_count": FROZEN_CONFIG_ANTIGRAVITY.official_command_invocations_must_remain,
        "result_scope": "aggregate_method_fold_qualification_only_no_routing_no_test_labels",
    }


__all__ = ("export_dashboard_antigravity",)
