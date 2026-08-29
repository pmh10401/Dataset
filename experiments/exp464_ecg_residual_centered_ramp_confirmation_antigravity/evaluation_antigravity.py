"""Frozen C6-only decision boundary for Exp464 Task 4."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp463AuthorityAntigravity,
    validate_exp463_authority_antigravity,
)
from .qualification_antigravity import (
    Exp464QualificationContractAntigravity,
    normalize_qualification_antigravity,
    validate_control_reproduction_antigravity,
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
            raise ValueError("status must be a non-empty string")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary candidate differs from frozen C6")
        if type(self.eligible_for_routing_research) is not bool:
            raise ValueError("eligible_for_routing_research must be bool")
        if type(self.primary_passed_count) is not int or not 0 <= self.primary_passed_count <= 10:
            raise ValueError("primary_passed_count must be between zero and ten")
        if self.primary_required_count != 10:
            raise ValueError("primary_required_count must equal ten")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be a tuple of non-empty strings")
        if self.routing_applied is not False:
            raise ValueError("Task 4 cannot apply routing")
        if self.test_labels_opened is not False:
            raise ValueError("Task 4 cannot open TEST labels")
        if self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("label_query_count must remain zero")
        if self.eligible_for_routing_research != (not self.failure_reasons):
            raise ValueError("eligibility must follow failure_reasons")


def evaluate_qualification_antigravity(
    qualification: Exp464QualificationContractAntigravity,
    authority: Exp463AuthorityAntigravity,
) -> QualificationDecisionAntigravity:
    validate_exp463_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(qualification, authority)
    primary_rows = tuple(
        row for row in normalized.fold_metrics if row.method == "C6"
    )
    if len(primary_rows) != 10:
        raise ValueError("primary candidate must contain exactly ten rows")
    primary_summary = next(
        summary for summary in normalized.summaries if summary.method == "C6"
    )
    reasons: list[str] = []
    if not primary_summary.passed or primary_summary.passed_count != 10:
        reasons.append("primary_candidate_not_10_of_10")
    for change_type in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"):
        if primary_summary.min_per_type_auc[change_type] < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"primary_{change_type}_auc_below_min")
    if primary_summary.min_mean_auc < FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min:
        reasons.append("primary_mean_auc_below_0.90")
    if primary_summary.max_normal_wilson_upper > FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max:
        reasons.append("primary_normal_wilson_upper_above_max")
    if not primary_summary.all_scores_finite:
        reasons.append("primary_scores_not_finite")
    if primary_summary.min_gradual_drift_auc < FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_min:
        reasons.append("primary_gradual_drift_min_below_0.85")
    if primary_summary.min_gradual_drift_auc <= FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_strictly_above_exp463:
        reasons.append("primary_gradual_drift_not_strictly_above_exp463_c0")
    if primary_summary.min_period_change_auc < FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min:
        reasons.append("primary_period_change_min_below_exp463_global_floor")
    if any("gradual_drift_auc_below_c0" in row.failure_reasons for row in primary_rows):
        reasons.append("primary_gradual_drift_below_c0_on_some_fold")
    if any("period_change_auc_below_c0" in row.failure_reasons for row in primary_rows):
        reasons.append("primary_period_change_below_c0_on_some_fold")
    for row in primary_rows:
        for reason in row.failure_reasons:
            if reason.startswith("historical_period_change_below_min:"):
                reasons.append(reason)
    reasons.extend(validate_control_reproduction_antigravity(normalized, authority))
    ordered_reasons = tuple(sorted(set(reasons)))
    eligible = not ordered_reasons
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
        failure_reasons=ordered_reasons,
        routing_applied=False,
        test_labels_opened=False,
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
    )


__all__ = ("QualificationDecisionAntigravity", "evaluate_qualification_antigravity")
