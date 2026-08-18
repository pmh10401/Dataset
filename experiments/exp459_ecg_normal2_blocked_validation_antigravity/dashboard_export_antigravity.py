"""Strict aggregate dashboard payload for Exp459 Task 5."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import ValidationDecisionAntigravity
from .evaluation_antigravity import (
    _qualification_for_decision_antigravity,
    exp458_predecessor_context_payload_antigravity,
    validate_frozen_validation_authority_antigravity,
    validate_validation_decision_semantics_antigravity,
)


def _dashboard_payload_antigravity(
    decision: object,
    fold_evidence: tuple[object, ...],
    seed_summaries: tuple[object, ...],
) -> dict[str, object]:
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "progress": "Task 5/6",
        "status": decision.status,
        "passed": decision.passed,
        "label_query_count": 0,
        "fold_records": [
            {
                "dataset": record.dataset,
                "seed": record.seed,
                "fold_index": record.fold_index,
                "normal_trials": record.normal_trials,
                "normal_alerts": record.normal_alerts,
                "normal_wilson_upper": record.normal_wilson_upper,
                "per_type_auc": {
                    name: record.per_type_auc[name]
                    for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
                },
                "pooled_auc": record.pooled_auc,
                "passed": record.passed,
            }
            for record in fold_evidence
        ],
        "seed_summaries": [
            {
                "dataset": summary.dataset,
                "seed": summary.seed,
                "fold_count": len(summary.fold_records),
                "passed": summary.passed,
            }
            for summary in seed_summaries
        ],
        "exp458_predecessor": exp458_predecessor_context_payload_antigravity(),
    }


def export_dashboard_antigravity(decision: ValidationDecisionAntigravity) -> dict[str, object]:
    value = validate_validation_decision_semantics_antigravity(decision)
    qualification = _qualification_for_decision_antigravity(value)
    return _dashboard_payload_antigravity(value, qualification.fold_evidence, qualification.seed_summaries)


def export_dashboard_from_authority_antigravity(authority: object) -> dict[str, object]:
    trusted = validate_frozen_validation_authority_antigravity(authority)
    return _dashboard_payload_antigravity(
        trusted.decision,
        trusted.fold_evidence,
        trusted.seed_summaries,
    )


__all__ = ("export_dashboard_antigravity", "export_dashboard_from_authority_antigravity")
