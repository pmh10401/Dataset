"""Scientific-only Task 4 dashboard payload for Exp460."""

from __future__ import annotations

from .contracts_antigravity import (
    Exp459FallbackAuthorityAntigravity,
    FallbackEvaluationAntigravity,
    validate_exp459_fallback_authority_antigravity,
    validate_fallback_evaluation_antigravity,
)


def _comparison_rows_antigravity(result: FallbackEvaluationAntigravity) -> list[dict[str, object]]:
    return [
        {
            "rule": row.rule,
            "route": row.route,
            "tp": row.tp,
            "fp": row.fp,
            "fn": row.fn,
            "tn": row.tn,
            "f1": row.f1,
        }
        for row in result.rule_results
    ]


def _failed_folds_antigravity(authority: Exp459FallbackAuthorityAntigravity) -> list[dict[str, object]]:
    return [
        {
            "seed": record.seed,
            "fold_index": record.fold_index,
            "period_change_auc": record.period_change_auc,
        }
        for record in authority.fold_records
        if not record.passed
    ]


def export_dashboard_antigravity(
    authority: Exp459FallbackAuthorityAntigravity,
    result: FallbackEvaluationAntigravity,
) -> dict[str, object]:
    trusted_authority = validate_exp459_fallback_authority_antigravity(authority)
    trusted_result = validate_fallback_evaluation_antigravity(result, trusted_authority)
    return {
        "comparison_rows": _comparison_rows_antigravity(trusted_result),
        "dataset": "ECG5000_normal_2",
        "experiment_id": "Exp460",
        "failed_folds": _failed_folds_antigravity(trusted_authority),
        "implementation_official": False,
        "joint_success": trusted_result.joint_success,
        "label_query_count": 0,
        "official_evaluate_invocations": 0,
        "official_inspect_invocations": 0,
        "predecessor_authority_digest": trusted_authority.frozen_authority_digest,
        "predecessor_experiment_id": "Exp459",
        "progress": "Task 4/6",
        "status": trusted_result.strict_status,
        "strict_route": trusted_result.strict_route,
        "strict_status": trusted_result.strict_status,
        "worst_period_change_auc": trusted_result.worst_period_change_auc,
    }


__all__ = ("export_dashboard_antigravity",)
