"""Exp461 Task 4 qualification decision."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp460PeriodAuthorityAntigravity,
    validate_exp460_period_authority_antigravity,
)
from .qualification_antigravity import (
    MethodQualificationAntigravity,
    normalize_qualification_antigravity,
    recompute_row_pass_antigravity,
)


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
        if type(self.status) is not str or not self.status:
            raise ValueError("status must be non-empty str")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary candidate differs")
        if type(self.eligible_for_routing_research) is not bool:
            raise ValueError("eligible_for_routing_research must be bool")
        if type(self.primary_passed_count) is not int or self.primary_passed_count < 0:
            raise ValueError("primary_passed_count must be non-negative int")
        if self.primary_required_count != (
            len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("primary_required_count differs")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be non-empty strings")
        if self.routing_applied is not False:
            raise ValueError("Task 4 must not apply routing")
        if self.test_labels_opened is not False:
            raise ValueError("Task 4 must not open TEST labels")
        if self.label_query_count != 0:
            raise ValueError("Task 4 label query count must remain zero")


def evaluate_qualification_antigravity(
    authority: Exp460PeriodAuthorityAntigravity,
    qualification: MethodQualificationAntigravity,
) -> QualificationDecisionAntigravity:
    trusted_authority = validate_exp460_period_authority_antigravity(authority)
    if type(qualification) is not MethodQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    summaries = normalized.summaries
    primary_summary = next(
        summary
        for summary in summaries
        if summary.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate
    )
    primary_rows = tuple(
        row
        for row in normalized.fold_metrics
        if row.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate
    )
    reasons: list[str] = []
    if not primary_summary.passed:
        reasons.append("primary_candidate_not_10_of_10")
    for row in primary_rows:
        _passed, row_reasons = recompute_row_pass_antigravity(row, trusted_authority)
        for reason in row_reasons:
            if reason.startswith("historical_period_change_not_strictly_improved"):
                reasons.append(reason)
    reasons = sorted(set(reasons))
    eligible = not reasons and primary_summary.passed
    return QualificationDecisionAntigravity(
        status=(
            "qualification_passed_ready_for_task5_reproducibility"
            if eligible
            else "qualification_failed_blocked_before_routing_research"
        ),
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        eligible_for_routing_research=eligible,
        primary_passed_count=primary_summary.passed_count,
        primary_required_count=primary_summary.fold_count,
        failure_reasons=tuple(reasons),
        routing_applied=False,
        test_labels_opened=False,
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )


__all__ = ("QualificationDecisionAntigravity", "evaluate_qualification_antigravity")
