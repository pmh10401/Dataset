"""Export only aggregate Exp455 routing results for dashboard use."""

from __future__ import annotations

from .contracts_antigravity import RoutingEvaluationAntigravity


def export_aggregate_dashboard_antigravity(evaluation: RoutingEvaluationAntigravity) -> dict[str, object]:
    """Return the fixed aggregate-only dashboard schema; never emit dataset evidence."""
    if type(evaluation) is not RoutingEvaluationAntigravity or len(evaluation.comparisons) != 4 or len(evaluation.family_fn_delta) != 2:
        raise ValueError("evaluation is invalid")
    return {
        "experiment_id": "Exp455",
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
