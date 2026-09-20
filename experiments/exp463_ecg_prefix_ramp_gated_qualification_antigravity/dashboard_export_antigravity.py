"""Aggregate-only dashboard projection for Exp463 Task 4."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp462AuthorityAntigravity,
    Exp463QualificationContractAntigravity,
    validate_exp462_authority_antigravity,
)
from .evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)
from .qualification_antigravity import normalize_qualification_antigravity


def _summary_payloads_antigravity(
    qualification: Exp463QualificationContractAntigravity,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for summary in qualification.summaries:
        payloads.append(
            {
                "method": summary.method,
                "fold_count": summary.fold_count,
                "passed_count": summary.passed_count,
                "failed_count": summary.failed_count,
                "min_per_type_auc": dict(summary.min_per_type_auc),
                "min_gradual_drift_auc": summary.min_gradual_drift_auc,
                "min_period_change_auc": summary.min_period_change_auc,
                "min_pooled_auc": summary.min_pooled_auc,
                "max_normal_wilson_upper": summary.max_normal_wilson_upper,
                "all_scores_finite": summary.all_scores_finite,
                "passed": summary.passed,
            }
        )
    return payloads


def export_dashboard_antigravity(
    authority: Exp462AuthorityAntigravity,
    qualification: Exp463QualificationContractAntigravity,
    decision: QualificationDecisionAntigravity,
) -> dict[str, object]:
    """Return ordered aggregate metrics without score vectors or source paths."""

    trusted_authority = validate_exp462_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    recomputed_decision = evaluate_qualification_antigravity(trusted_authority, normalized)
    if decision != recomputed_decision:
        raise ValueError("dashboard decision differs from evidence-derived decision")
    method_summaries = _summary_payloads_antigravity(normalized)
    primary_rows = tuple(
        row
        for row in normalized.fold_metrics
        if row.method == "M2"
    )
    return {
        "experiment": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
        "stage": "Task4 implemented pending external review / Task5 next",
        "primary_candidate": FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        "primary_status": decision.status,
        "method_order": list(FROZEN_CONFIG_ANTIGRAVITY.method_order),
        "method_summaries": method_summaries,
        "failed_fold_count": sum(int(summary["failed_count"]) for summary in method_summaries),
        "historical_improvements": [
            {
                "seed": historical.seed,
                "fold_index": historical.fold_index,
                "previous_period_change_auc": historical.period_change_auc,
                "candidate_period_change_auc": next(
                    row.per_type_auc["period_change"]
                    for row in primary_rows
                    if row.seed == historical.seed and row.fold_index == historical.fold_index
                ),
            }
            for historical in trusted_authority.historical_period_folds
        ],
        "decision": {
            "eligible_for_routing_research": decision.eligible_for_routing_research,
            "status": decision.status,
            "failure_reasons": list(decision.failure_reasons),
            "routing_applied": decision.routing_applied,
        },
        "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        "official_command_count": FROZEN_CONFIG_ANTIGRAVITY.official_command_invocations_must_remain,
        "result_scope": "aggregate_method_fold_qualification_only_no_label_data",
    }


__all__ = ("export_dashboard_antigravity",)
