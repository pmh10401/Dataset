"""Export the strict aggregate/TRAIN-evidence dashboard payload for Exp458."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import EvaluationResultAntigravity
from .evaluation_antigravity import (
    _FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY,
    _FINAL_FN_MAX_ANTIGRAVITY,
    _FINAL_FP_MAX_ANTIGRAVITY,
    _evidence_for_result_antigravity,
    validate_evaluation_result_semantics_antigravity,
)


_COMPARISONS_ANTIGRAVITY = (
    "always_baseline",
    "family_wide_local_gap",
    "oracle_per_dataset",
    "normal2_rank_router",
)
_CRITERIA_ANTIGRAVITY = (
    ("final_fn", "<=", _FINAL_FN_MAX_ANTIGRAVITY),
    ("final_fp", "<=", _FINAL_FP_MAX_ANTIGRAVITY),
    ("final_f1", ">", _FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY),
)


def _validate_result_antigravity(value: object) -> EvaluationResultAntigravity:
    if type(value) is not EvaluationResultAntigravity:
        raise ValueError("dashboard input must be an Exp458 evaluation")
    result = validate_evaluation_result_semantics_antigravity(value)
    if tuple(row.name for row in result.comparisons) != _COMPARISONS_ANTIGRAVITY:
        raise ValueError("dashboard comparison order differs")
    if tuple((row.metric, row.operator, row.threshold) for row in result.criteria) != _CRITERIA_ANTIGRAVITY:
        raise ValueError("dashboard criteria differ")
    if tuple(row.dataset for row in result.decisions) != (FROZEN_CONFIG_ANTIGRAVITY.dataset,):
        raise ValueError("dashboard route identity differs")
    final = result.comparisons[-1]
    expected_actuals = (final.confusion.fn, final.confusion.fp, final.f1)
    expected_passed = (
        final.confusion.fn <= _FINAL_FN_MAX_ANTIGRAVITY,
        final.confusion.fp <= _FINAL_FP_MAX_ANTIGRAVITY,
        final.f1 > _FINAL_F1_STRICTLY_ABOVE_ANTIGRAVITY,
    )
    if tuple(row.actual for row in result.criteria) != expected_actuals or tuple(row.passed for row in result.criteria) != expected_passed:
        raise ValueError("dashboard criteria outcomes differ")
    passed = all(expected_passed)
    status = (
        "routing_candidate_qualified_for_external_validation"
        if passed
        else "routing_candidate_not_qualified_for_external_validation"
    )
    if result.passed is not passed or result.status != status:
        raise ValueError("dashboard status differs")
    return result


def export_dashboard_antigravity(result: EvaluationResultAntigravity) -> dict[str, object]:
    value = _validate_result_antigravity(result)
    _authority, qualification, _feature = _evidence_for_result_antigravity(value)
    decision = value.decisions[0]
    return {
        "experiment_id": "Exp458",
        "status": value.status,
        "passed": value.passed,
        "rank_evidence": [
            {
                "dataset": FROZEN_CONFIG_ANTIGRAVITY.dataset,
                "seed": seed.seed,
                "normal_trials": seed.normal_trials,
                "normal_alerts": seed.normal_alerts,
                "normal_wilson_upper": seed.normal_wilson_upper,
                "per_type_auc": dict(seed.per_type_auc),
                "pooled_auc": seed.pooled_auc,
                "passed": seed.passed,
            }
            for seed in qualification.seed_evidence
        ],
        "normal2_route": {
            "dataset": decision.dataset,
            "rank_gate": decision.rank_gate,
            "structure_gate": decision.structure_gate,
            "route": decision.route,
        },
        "methods": [
            {
                "method": row.name,
                "tp": row.confusion.tp,
                "fp": row.confusion.fp,
                "fn": row.confusion.fn,
                "tn": row.confusion.tn,
                "f1": row.f1,
            }
            for row in value.comparisons
        ],
        "criteria": [
            {
                "metric": row.metric,
                "operator": row.operator,
                "threshold": row.threshold,
                "actual": row.actual,
                "passed": row.passed,
            }
            for row in value.criteria
        ],
    }


__all__ = ("export_dashboard_antigravity",)
