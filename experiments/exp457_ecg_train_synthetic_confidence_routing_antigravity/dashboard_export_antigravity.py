"""Export the strict aggregate/TRAIN-gate Exp457 dashboard allowlist."""

from __future__ import annotations

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import (
    _f1_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import EvaluationResultAntigravity
from .evaluation_antigravity import validate_evaluation_result_semantics_antigravity


_COMPARISONS_ANTIGRAVITY = (
    "always_baseline",
    "family_wide_local_gap",
    "oracle_per_dataset",
    "ecg_train_confidence_router",
)
_FAMILIES_ANTIGRAVITY = ("ECG5000", "ChlorineConcentration")
_CRITERIA_ANTIGRAVITY = (
    ("final_fn", "<=", FROZEN_CONFIG_ANTIGRAVITY.final_fn_max),
    ("final_fp", "<=", FROZEN_CONFIG_ANTIGRAVITY.final_fp_max),
    ("final_f1", ">", FROZEN_CONFIG_ANTIGRAVITY.final_f1_strictly_above),
)


def _validate_result_antigravity(value: object) -> EvaluationResultAntigravity:
    if type(value) is not EvaluationResultAntigravity:
        raise ValueError("dashboard input must be an Exp457 evaluation")
    validate_evaluation_result_semantics_antigravity(value)
    if tuple(row.name for row in value.comparisons) != _COMPARISONS_ANTIGRAVITY:
        raise ValueError("dashboard comparison order differs")
    for row in value.comparisons:
        confusion = row.confusion
        if any(
            type(item) is not int or item < 0
            for item in (confusion.tp, confusion.fp, confusion.fn, confusion.tn)
        ):
            raise ValueError("dashboard confusion is invalid")
        if type(row.f1) is not float or row.f1 != float(_f1_antigravity(confusion)):
            raise ValueError("dashboard F1 is invalid")
    if tuple(row.dataset for row in value.decisions) != FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("dashboard ECG order differs")
    if tuple(row.family for row in value.family_metrics) != _FAMILIES_ANTIGRAVITY:
        raise ValueError("dashboard family order differs")
    if tuple((row.metric, row.operator, row.threshold) for row in value.criteria) != _CRITERIA_ANTIGRAVITY:
        raise ValueError("dashboard criteria differ")
    final = value.comparisons[-1]
    actuals = (final.confusion.fn, final.confusion.fp, final.f1)
    outcomes = (
        final.confusion.fn <= FROZEN_CONFIG_ANTIGRAVITY.final_fn_max,
        final.confusion.fp <= FROZEN_CONFIG_ANTIGRAVITY.final_fp_max,
        final.f1 > FROZEN_CONFIG_ANTIGRAVITY.final_f1_strictly_above,
    )
    if tuple(row.actual for row in value.criteria) != actuals or tuple(
        row.passed for row in value.criteria
    ) != outcomes:
        raise ValueError("dashboard criteria outcomes differ")
    passed = all(outcomes)
    expected_status = (
        "routing_candidate_qualified_for_external_validation"
        if passed
        else "routing_candidate_not_qualified_for_external_validation"
    )
    if value.passed is not passed or value.status != expected_status:
        raise ValueError("dashboard decision differs")
    return value


def export_dashboard_antigravity(result: EvaluationResultAntigravity) -> dict[str, object]:
    """Return only the preregistered aggregate and gate/route fields."""
    value = _validate_result_antigravity(result)
    return {
        "experiment_id": "Exp457",
        "status": value.status,
        "passed": value.passed,
        "ecg_routes": [
            {
                "dataset": row.dataset,
                "train_gate": row.train_gate,
                "structure_gate": row.structure_gate,
                "route": row.route,
            }
            for row in value.decisions
        ],
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
        "family_metrics": [
            {
                "family": row.family,
                "fn_delta": row.fn_delta,
                "fp_delta": row.fp_delta,
            }
            for row in value.family_metrics
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
