"""Frozen decision boundary for the blocked Exp463 qualification."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp462AuthorityAntigravity,
    Exp463QualificationContractAntigravity,
    validate_exp462_authority_antigravity,
)
from .qualification_antigravity import (
    normalize_qualification_antigravity,
    recompute_row_pass_antigravity,
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
            raise ValueError("primary candidate differs from frozen M2")
        if type(self.eligible_for_routing_research) is not bool:
            raise ValueError("eligible_for_routing_research must be bool")
        if type(self.primary_passed_count) is not int or not 0 <= self.primary_passed_count <= 10:
            raise ValueError("primary_passed_count must be between zero and ten")
        if self.primary_required_count != 10:
            raise ValueError("primary_required_count must equal ten")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be non-empty strings")
        if self.routing_applied is not False:
            raise ValueError("Task 4 cannot apply routing")
        if self.test_labels_opened is not False:
            raise ValueError("Task 4 cannot open label data")
        if self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("label query count must remain zero")
        if self.eligible_for_routing_research != (not self.failure_reasons):
            raise ValueError("eligibility must follow frozen failure reasons")


def evaluate_qualification_antigravity(
    authority: Exp462AuthorityAntigravity,
    qualification: Exp463QualificationContractAntigravity,
) -> QualificationDecisionAntigravity:
    """Recompute all metrics and decide only the preregistered M2 candidate."""

    trusted_authority = validate_exp462_authority_antigravity(authority)
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    primary_rows = tuple(
        row
        for row in normalized.fold_metrics
        if row.method == "M2"
    )
    if len(primary_rows) != 10:
        raise ValueError("primary candidate must contain exactly ten rows")
    primary_summary = next(
        summary
        for summary in normalized.summaries
        if summary.method == "M2"
    )
    reasons: list[str] = []
    if not primary_summary.passed or primary_summary.passed_count != 10:
        reasons.append("primary_candidate_not_10_of_10")
    if primary_summary.min_gradual_drift_auc < FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_min:
        reasons.append("primary_gradual_drift_min_below_0.85")
    if primary_summary.min_gradual_drift_auc <= FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_strictly_above_exp462:
        reasons.append("primary_gradual_drift_not_strictly_improved_over_exp462")
    if primary_summary.min_period_change_auc < FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min:
        reasons.append("primary_period_change_min_below_exp462_global_floor")
    if primary_summary.max_normal_wilson_upper > FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max:
        reasons.append("primary_normal_wilson_upper_above_max")
    if not primary_summary.all_scores_finite:
        reasons.append("primary_scores_not_finite")
    for historical in trusted_authority.historical_period_folds:
        row = next(
            (
                candidate
                for candidate in primary_rows
                if candidate.seed == historical.seed and candidate.fold_index == historical.fold_index
            ),
            None,
        )
        if row is None or row.per_type_auc["period_change"] < historical.period_change_auc:
            reasons.append(f"historical_period_change_below_min:{historical.seed}:{historical.fold_index}")
    for row in primary_rows:
        _passed, row_reasons = recompute_row_pass_antigravity(row, trusted_authority)
        reasons.extend(row_reasons)
    reasons.extend(validate_control_reproduction_antigravity(normalized, trusted_authority))
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
