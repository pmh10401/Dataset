"""Export only aggregate Exp456 routing results for dashboard use."""

from __future__ import annotations

from fractions import Fraction

from .contracts_antigravity import RoutingEvaluationAntigravity


_COMPARISON_NAMES_ANTIGRAVITY = ("always_baseline", "family_wide_local_gap", "oracle_per_dataset", "lodo_router")
_FAMILIES_ANTIGRAVITY = ("ECG5000", "ChlorineConcentration")


def _f1_antigravity(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(Fraction(0, 1) if denominator == 0 else Fraction(2 * tp, denominator))


def _validate_evaluation_antigravity(evaluation: RoutingEvaluationAntigravity) -> None:
    if type(evaluation) is not RoutingEvaluationAntigravity or type(evaluation.status) is not str or evaluation.status not in {"routing_candidate_qualified_for_external_validation", "routing_candidate_not_qualified_for_external_validation"} or type(evaluation.passed) is not bool:
        raise ValueError("dashboard evaluation is invalid")
    if tuple(row.name for row in evaluation.comparisons) != _COMPARISON_NAMES_ANTIGRAVITY:
        raise ValueError("dashboard comparison order is invalid")
    for row in evaluation.comparisons:
        confusion = row.confusion
        if any(type(value) is not int or value < 0 for value in (confusion.tp, confusion.fp, confusion.fn, confusion.tn)) or type(row.f1) is not float or row.f1 != _f1_antigravity(confusion.tp, confusion.fp, confusion.fn):
            raise ValueError("dashboard comparison value is invalid")
    if tuple(family for family, _delta in evaluation.family_fn_delta) != _FAMILIES_ANTIGRAVITY or any(type(delta) is not int for _family, delta in evaluation.family_fn_delta):
        raise ValueError("dashboard family values are invalid")


def export_aggregate_dashboard_antigravity(evaluation: RoutingEvaluationAntigravity) -> dict[str, object]:
    """Return the fixed aggregate-only dashboard schema; never emit dataset evidence."""
    _validate_evaluation_antigravity(evaluation)
    return {
        "experiment_id": "Exp456",
        "status": evaluation.status,
        "completed_folds": 8,
        "methods": [
            {"method": comparison.name, "fn": comparison.confusion.fn, "fp": comparison.confusion.fp, "f1": comparison.f1}
            for comparison in evaluation.comparisons
        ],
        "family_fn_delta": [
            {"family": family, "fn_delta": delta} for family, delta in evaluation.family_fn_delta
        ],
        "passed": evaluation.passed,
    }


__all__ = ["export_aggregate_dashboard_antigravity"]

