"""Task 3 rule comparison and strict fallback evaluation for Exp460."""

from __future__ import annotations

from .contracts_antigravity import (
    ConfusionAntigravity,
    Exp459FallbackAuthorityAntigravity,
    FallbackEvaluationAntigravity,
    FallbackRuleComparisonAntigravity,
    _f1_antigravity,
    _status_antigravity,
    _task3_snapshot_from_authority_antigravity,
    validate_fallback_evaluation_antigravity,
)
from .fallback_antigravity import strict_route_antigravity


def _row_antigravity(
    rule: str,
    route: str,
    confusion: ConfusionAntigravity,
) -> FallbackRuleComparisonAntigravity:
    return FallbackRuleComparisonAntigravity(
        rule=rule,
        route=route,
        tp=confusion.tp,
        fp=confusion.fp,
        fn=confusion.fn,
        tn=confusion.tn,
        f1=_f1_antigravity(confusion.tp, confusion.fp, confusion.fn),
    )


def compare_fallback_rules_antigravity(
    authority: Exp459FallbackAuthorityAntigravity,
) -> FallbackEvaluationAntigravity:
    """Recompute four ordered rule comparisons and the strict final decision."""

    snapshot = _task3_snapshot_from_authority_antigravity(authority)
    passed_record_count = snapshot["passed_record_count"]
    all_records_passed = snapshot["all_records_passed"]
    worst_period_change_auc = snapshot["worst_period_change_auc"]
    baseline_confusion = snapshot["baseline_confusion"]
    candidate_confusion = snapshot["candidate_confusion"]
    assert type(passed_record_count) is int
    assert type(all_records_passed) is bool
    assert type(worst_period_change_auc) is float
    assert type(baseline_confusion) is ConfusionAntigravity
    assert type(candidate_confusion) is ConfusionAntigravity

    strict_route = strict_route_antigravity(authority)
    majority_route = "candidate" if passed_record_count >= 8 else "baseline"
    rule_results = (
        _row_antigravity("always_baseline", "baseline", baseline_confusion),
        _row_antigravity("always_candidate", "candidate", candidate_confusion),
        _row_antigravity(
            "majority_8_of_10",
            majority_route,
            candidate_confusion if majority_route == "candidate" else baseline_confusion,
        ),
        _row_antigravity(
            "strict_10_of_10",
            strict_route,
            candidate_confusion if strict_route == "candidate" else baseline_confusion,
        ),
    )
    strict_row = rule_results[3]
    baseline_row = rule_results[0]
    joint_success = (
        strict_route == "candidate"
        and all_records_passed
        and worst_period_change_auc >= 0.85
        and strict_row.fn < baseline_row.fn
        and strict_row.fp <= baseline_row.fp
        and strict_row.f1 > baseline_row.f1
    )
    result = FallbackEvaluationAntigravity(
        rule_results=rule_results,
        strict_route=strict_route,
        strict_status=_status_antigravity(strict_route, joint_success),
        joint_success=joint_success,
        all_records_passed=all_records_passed,
        passed_record_count=passed_record_count,
        worst_period_change_auc=worst_period_change_auc,
    )
    return validate_fallback_evaluation_antigravity(result, authority)


__all__ = ("compare_fallback_rules_antigravity",)
