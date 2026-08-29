"""Pair-repair-only decision boundary for Exp470 Task 4."""

from __future__ import annotations

from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp470QualificationAntigravity


@dataclass(frozen=True, slots=True)
class QualificationDecisionAntigravity:
    passed: bool
    eligible_for_private_reproducibility: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.eligible_for_private_reproducibility) is not bool:
            raise ValueError("decision state must use exact bool")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str or not reason for reason in self.failure_reasons
        ) or self.failure_reasons != tuple(dict.fromkeys(self.failure_reasons)):
            raise ValueError("failure_reasons must be unique nonempty strings in fixed order")
        expected = not self.failure_reasons
        if self.passed is not expected or self.eligible_for_private_reproducibility is not expected:
            raise ValueError("decision eligibility differs from failure reasons")


def evaluate_repair_gates_antigravity(qualification: object) -> QualificationDecisionAntigravity:
    """Use only preregistered pair-repair, fallback, and finite gates; AUC is observational."""
    if type(qualification) is not Exp470QualificationAntigravity:
        raise ValueError("decision requires exact Exp470QualificationAntigravity")
    reasons: list[str] = []
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if qualification.period_fixed[seed] < 1:
            reasons.append(f"seed_{seed}_period_fixed_below_1")
        if qualification.period_broken[seed] != 0:
            reasons.append(f"seed_{seed}_period_broken_not_zero")
    for name in FROZEN_CONFIG_ANTIGRAVITY.change_types[:3]:
        if qualification.protected_broken[name] != 0:
            reasons.append(f"{name}_broken_not_zero")
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if qualification.normal_confirmations[seed] != 0:
            reasons.append(f"seed_{seed}_normal_confirmations_not_zero")
    if not qualification.fallback_exact:
        reasons.append("fallback_not_exact")
    if not qualification.finite:
        reasons.append("nonfinite_scores")
    failures = tuple(reasons)
    passed = not failures
    return QualificationDecisionAntigravity(passed=passed, eligible_for_private_reproducibility=passed, failure_reasons=failures)


__all__ = ("QualificationDecisionAntigravity", "evaluate_repair_gates_antigravity")
