"""Pure frozen C9 qualification decision for Exp466."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp466QualificationAntigravity


_PASS_STATUS_ANTIGRAVITY = "qualification_passed_awaiting_exp467_design"
_FAIL_STATUS_ANTIGRAVITY = "qualification_failed_blocked_before_fn_validation"


@dataclass(frozen=True, slots=True)
class QualificationDecisionAntigravity:
    status: str
    primary_candidate: str
    eligible_for_exp467_design: bool
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
            raise ValueError("Exp466 primary candidate must be C9")
        if self.primary_required_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
            raise ValueError("primary required count differs from frozen ten records")
        if (
            type(self.primary_passed_count) is not int
            or not 0 <= self.primary_passed_count <= self.primary_required_count
        ):
            raise ValueError("primary passed count is invalid")
        for name in (
            "eligible_for_exp467_design",
            "routing_applied",
            "champion_modified",
            "pending_modified",
            "promotion_applied",
            "test_labels_opened",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be bool")
        if any(
            (
                self.routing_applied,
                self.champion_modified,
                self.pending_modified,
                self.promotion_applied,
                self.test_labels_opened,
            ),
        ):
            raise ValueError("Exp466 qualification decision cannot mutate external state")
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("Exp466 qualification decision cannot query TEST labels")
        if (
            type(self.failure_reasons) is not tuple
            or any(type(reason) is not str or not reason for reason in self.failure_reasons)
            or len(set(self.failure_reasons)) != len(self.failure_reasons)
        ):
            raise ValueError("failure_reasons must be a unique immutable string tuple")
        eligible = len(self.failure_reasons) == 0
        if self.eligible_for_exp467_design is not eligible:
            raise ValueError("eligibility differs from failure reasons")
        if eligible and self.primary_passed_count != self.primary_required_count:
            raise ValueError("eligible decision requires ten passing folds")
        expected_status = _PASS_STATUS_ANTIGRAVITY if eligible else _FAIL_STATUS_ANTIGRAVITY
        if self.status != expected_status:
            raise ValueError("status differs from frozen eligibility decision")


def evaluate_qualification_antigravity(
    q: Exp466QualificationAntigravity,
) -> QualificationDecisionAntigravity:
    """Read only frozen scalar rows; never load data, rescore, publish, or route."""

    if type(q) is not Exp466QualificationAntigravity:
        raise ValueError("decision requires exact Exp466QualificationAntigravity")
    passed_count = sum(row.passed for row in q.fold_results)
    reasons: list[str] = []
    if passed_count != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
        reasons.append("c9_not_10_of_10")
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
        eligible_for_exp467_design=eligible,
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


__all__ = ("QualificationDecisionAntigravity", "evaluate_qualification_antigravity")
