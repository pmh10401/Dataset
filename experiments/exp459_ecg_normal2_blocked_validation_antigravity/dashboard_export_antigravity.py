"""Strict aggregate dashboard payload for Exp459 Task 4."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import ValidationDecisionAntigravity
from .evaluation_antigravity import (
    EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
    _qualification_for_decision_antigravity,
    validate_validation_decision_semantics_antigravity,
)


def export_dashboard_antigravity(decision: ValidationDecisionAntigravity) -> dict[str, object]:
    value = validate_validation_decision_semantics_antigravity(decision)
    qualification = _qualification_for_decision_antigravity(value)
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "progress": "Task 4/6",
        "status": value.status,
        "passed": value.passed,
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
            for record in qualification.fold_evidence
        ],
        "seed_summaries": [
            {
                "dataset": summary.dataset,
                "seed": summary.seed,
                "fold_count": len(summary.fold_records),
                "passed": summary.passed,
            }
            for summary in qualification.seed_summaries
        ],
        "exp458_predecessor": dict(EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY),
    }


__all__ = ("export_dashboard_antigravity",)
