"""C8-only decision boundary for Exp465 Task 4."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .qualification_antigravity import Exp465QualificationResultAntigravity, recompute_qualification_antigravity


@dataclass(frozen=True, slots=True)
class QualificationDecisionAntigravity:
    status: str
    primary_candidate: str
    eligible_for_routing_research: bool
    primary_passed_count: int
    primary_required_count: int
    failure_reasons: tuple[str, ...]
    routing_applied: bool
    test_labels_opened: bool
    label_query_count: int

    def __post_init__(self) -> None:
        if self.primary_candidate != "C8":
            raise ValueError("Exp465 primary candidate must be C8")
        if type(self.eligible_for_routing_research) is not bool or type(self.routing_applied) is not bool:
            raise ValueError("decision flags must be bool")
        if self.routing_applied:
            raise ValueError("Task 4 cannot apply routing")
        if self.test_labels_opened is not False or self.label_query_count != 0:
            raise ValueError("Task 4 cannot open TEST labels")
        if self.primary_required_count != FROZEN_CONFIG_ANTIGRAVITY.required_stability_records:
            raise ValueError("required record count differs from frozen contract")
        if not 0 <= self.primary_passed_count <= self.primary_required_count:
            raise ValueError("primary passed count is invalid")
        if type(self.failure_reasons) is not tuple or any(type(item) is not str or not item for item in self.failure_reasons):
            raise ValueError("failure reasons must be non-empty strings")
        if self.eligible_for_routing_research != (not self.failure_reasons):
            raise ValueError("decision eligibility differs from failure reasons")


def evaluate_qualification_antigravity(
    qualification: Exp465QualificationResultAntigravity,
) -> QualificationDecisionAntigravity:
    """Derive the frozen C8 decision from replayed raw score evidence."""
    replayed = recompute_qualification_antigravity(qualification)
    c8 = next(summary for summary in replayed.summaries if summary.method == "C8")
    reasons: list[str] = []
    if c8.passed_count != FROZEN_CONFIG_ANTIGRAVITY.required_stability_records:
        reasons.append("c8_not_10_of_10")
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if replayed.seed_normal_confirmations[seed] != 0:
            reasons.append(f"seed_{seed}_normal_confirmations_not_zero")
    for row in replayed.rows:
        if row.c8_gradual_auc < row.c0_gradual_auc:
            reasons.append("c8_gradual_auc_below_c0")
        if row.c8_period_auc < row.c0_period_auc:
            reasons.append("c8_period_auc_below_c0")
        if not row.all_scores_finite:
            reasons.append("c8_scores_not_finite")
    failure_reasons = tuple(sorted(set(reasons)))
    eligible = not failure_reasons
    return QualificationDecisionAntigravity(
        status=("qualification_passed_ready_for_task5_reproducibility" if eligible else "qualification_failed_blocked_before_routing_research"),
        primary_candidate="C8",
        eligible_for_routing_research=eligible,
        primary_passed_count=c8.passed_count,
        primary_required_count=FROZEN_CONFIG_ANTIGRAVITY.required_stability_records,
        failure_reasons=failure_reasons,
        routing_applied=False,
        test_labels_opened=False,
        label_query_count=0,
    )


__all__ = ("QualificationDecisionAntigravity", "evaluate_qualification_antigravity")
