"""Pure scalar-only Task 4 qualification boundary for Exp468."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp468QualificationAntigravity


_PASS_STATUS_ANTIGRAVITY = (
    "qualification_gate_passed_awaiting_private_reproducibility"
)
_FAIL_STATUS_ANTIGRAVITY = "qualification_failed_blocked_before_private_run"


@dataclass(frozen=True, slots=True)
class QualificationDecisionAntigravity:
    status: str
    primary_candidate: str
    eligible_for_private_reproducibility: bool
    primary_passed_count: int
    primary_required_count: int
    failure_reasons: tuple[str, ...]
    routing_applied: bool
    champion_modified: bool
    pending_modified: bool
    promotion_applied: bool
    test_labels_opened: bool
    label_query_count: int

    def __post_init__(self) -> None:
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("Exp468 primary candidate must be C10")
        if self.primary_required_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
            raise ValueError("primary required count differs from frozen ten records")
        if (
            type(self.primary_passed_count) is not int
            or not 0 <= self.primary_passed_count <= self.primary_required_count
        ):
            raise ValueError("primary passed count is invalid")
        for name in (
            "eligible_for_private_reproducibility",
            "routing_applied",
            "champion_modified",
            "pending_modified",
            "promotion_applied",
            "test_labels_opened",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be exact bool")
        if any(
            (
                self.routing_applied,
                self.champion_modified,
                self.pending_modified,
                self.promotion_applied,
                self.test_labels_opened,
            ),
        ):
            raise ValueError("Task 4 qualification cannot mutate external state")
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("Task 4 qualification cannot query TEST labels")
        if (
            type(self.failure_reasons) is not tuple
            or any(type(reason) is not str or not reason for reason in self.failure_reasons)
            or len(set(self.failure_reasons)) != len(self.failure_reasons)
        ):
            raise ValueError("failure_reasons must be a unique immutable string tuple")
        eligible = len(self.failure_reasons) == 0
        if self.eligible_for_private_reproducibility is not eligible:
            raise ValueError("private reproducibility eligibility differs from failure reasons")
        if eligible and self.primary_passed_count != self.primary_required_count:
            raise ValueError("eligible decision requires ten passing folds")
        expected_status = _PASS_STATUS_ANTIGRAVITY if eligible else _FAIL_STATUS_ANTIGRAVITY
        if self.status != expected_status:
            raise ValueError("status differs from the frozen Task 4 boundary")


def evaluate_qualification_antigravity(
    q: Exp468QualificationAntigravity,
) -> QualificationDecisionAntigravity:
    """Read exact scalar rows only; never load, rescore, publish, route, or promote."""

    if type(q) is not Exp468QualificationAntigravity:
        raise ValueError("decision requires exact Exp468QualificationAntigravity")
    passed_count = sum(row.passed for row in q.fold_results)
    reasons: list[str] = []
    if passed_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
        reasons.append("c10_not_10_of_10")
    for row in q.fold_results:
        for reason in row.failure_reasons:
            if reason not in reasons:
                reasons.append(reason)
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if q.normal_confirmations[seed] != 0:
            reason = f"seed_{seed}_normal_confirmations_not_zero"
            if reason not in reasons:
                reasons.append(reason)
    failure_reasons = tuple(reasons)
    eligible = len(failure_reasons) == 0
    return QualificationDecisionAntigravity(
        status=_PASS_STATUS_ANTIGRAVITY if eligible else _FAIL_STATUS_ANTIGRAVITY,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.primary_candidate,
        eligible_for_private_reproducibility=eligible,
        primary_passed_count=passed_count,
        primary_required_count=FROZEN_CONFIG_ANTIGRAVITY.required_fold_results,
        failure_reasons=failure_reasons,
        routing_applied=False,
        champion_modified=False,
        pending_modified=False,
        promotion_applied=False,
        test_labels_opened=False,
        label_query_count=0,
    )


__all__ = (
    "QualificationDecisionAntigravity",
    "evaluate_qualification_antigravity",
)
