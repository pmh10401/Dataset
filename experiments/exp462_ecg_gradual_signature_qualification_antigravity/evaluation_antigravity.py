"""Decision boundary for the TRAIN-only Exp462 qualification."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp461GradualAuthorityAntigravity,
    validate_exp461_gradual_authority_antigravity,
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
            raise ValueError("status must be a non-empty string")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary candidate differs")
        if type(self.eligible_for_routing_research) is not bool:
            raise ValueError("eligible_for_routing_research must be bool")
        if type(self.primary_passed_count) is not int or self.primary_passed_count < 0:
            raise ValueError("primary_passed_count must be non-negative int")
        expected_count = len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
        if self.primary_required_count != expected_count:
            raise ValueError("primary_required_count differs from the frozen ten records")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be non-empty strings")
        if self.routing_applied is not False:
            raise ValueError("Task 4 must not apply routing")
        if self.test_labels_opened is not False:
            raise ValueError("Task 4 must not open TEST labels")
        if self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("Task 4 label query count must remain zero")


def evaluate_qualification_antigravity(
    authority: Exp461GradualAuthorityAntigravity,
    qualification: MethodQualificationAntigravity,
) -> QualificationDecisionAntigravity:
    """Normalize score evidence and evaluate all frozen primary gates."""

    trusted_authority = validate_exp461_gradual_authority_antigravity(authority)
    if type(qualification) is not MethodQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    normalized = normalize_qualification_antigravity(trusted_authority, qualification)
    primary_rows = tuple(
        row
        for row in normalized.fold_metrics
        if row.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate
    )
    primary_summary = next(
        summary
        for summary in normalized.summaries
        if summary.method == FROZEN_CONFIG_ANTIGRAVITY.primary_candidate
    )
    reasons: list[str] = []
    if not primary_summary.passed:
        reasons.append("primary_candidate_not_10_of_10")
    gradual_min = min(row.per_type_auc["gradual_drift"] for row in primary_rows)
    period_min = min(row.per_type_auc["period_change"] for row in primary_rows)
    if gradual_min < FROZEN_CONFIG_ANTIGRAVITY.gradual_drift_auc_min:
        reasons.append("primary_gradual_drift_min_below_0.85")
    if gradual_min <= FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc:
        reasons.append("primary_gradual_drift_not_strictly_improved_over_exp461")
    if period_min < FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_global_min:
        reasons.append("primary_period_change_min_below_exp461_global_floor")
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
    ordered_reasons = tuple(sorted(set(reasons)))
    eligible = not ordered_reasons and primary_summary.passed
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
