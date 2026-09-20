"""Pair-repair-only decision boundary for Exp471 Task 4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


@dataclass(frozen=True, slots=True)
class QualificationDecisionAntigravity:
    passed: bool
    eligible_for_private_reproducibility: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.passed) is not bool or type(self.eligible_for_private_reproducibility) is not bool:
            raise ValueError("decision state must use exact bool")
        if (
            type(self.failure_reasons) is not tuple
            or any(type(reason) is not str or not reason for reason in self.failure_reasons)
            or self.failure_reasons != tuple(dict.fromkeys(self.failure_reasons))
        ):
            raise ValueError("failure_reasons must be unique nonempty strings in fixed order")
        expected = not self.failure_reasons
        if self.passed is not expected or self.eligible_for_private_reproducibility is not expected:
            raise ValueError("decision eligibility differs from failure reasons")


def _exact_count_mapping_antigravity(
    values: object,
    expected_keys: tuple[object, ...],
    name: str,
) -> Mapping[object, int]:
    if not isinstance(values, Mapping) or tuple(values) != expected_keys:
        raise ValueError(f"{name} must preserve frozen key order")
    if any(type(values[key]) is not int or values[key] < 0 for key in expected_keys):
        raise ValueError(f"{name} values must be non-negative exact integers")
    return values


def evaluate_repair_gates_antigravity(
    *,
    period_fixed: object,
    period_broken: object,
    protected_broken: object,
    normal_confirmations: object,
    fallback_exact: object,
    finite: object,
) -> QualificationDecisionAntigravity:
    """Apply only preregistered pair-repair gates; AUC remains observational."""
    fixed = _exact_count_mapping_antigravity(
        period_fixed, FROZEN_CONFIG_ANTIGRAVITY.seeds, "period_fixed"
    )
    broken = _exact_count_mapping_antigravity(
        period_broken, FROZEN_CONFIG_ANTIGRAVITY.seeds, "period_broken"
    )
    protected = _exact_count_mapping_antigravity(
        protected_broken, FROZEN_CONFIG_ANTIGRAVITY.change_types[:3], "protected_broken"
    )
    normal = _exact_count_mapping_antigravity(
        normal_confirmations, FROZEN_CONFIG_ANTIGRAVITY.seeds, "normal_confirmations"
    )
    if type(fallback_exact) is not bool or type(finite) is not bool:
        raise ValueError("fallback_exact and finite must be exact bool")

    reasons: list[str] = []
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        if fixed[seed] < 1:
            reasons.append(f"period_fixed_zero:{seed}")
        if broken[seed] != 0:
            reasons.append(f"period_broken:{seed}")
        if normal[seed] != 0:
            reasons.append(f"normal_confirmation:{seed}")
    for change in FROZEN_CONFIG_ANTIGRAVITY.change_types[:3]:
        if protected[change] != 0:
            reasons.append(f"protected_broken:{change}")
    if not fallback_exact:
        reasons.append("fallback_not_exact")
    if not finite:
        reasons.append("nonfinite")
    failures = tuple(reasons)
    passed = not failures
    return QualificationDecisionAntigravity(
        passed=passed,
        eligible_for_private_reproducibility=passed,
        failure_reasons=failures,
    )


__all__ = ("QualificationDecisionAntigravity", "evaluate_repair_gates_antigravity")
