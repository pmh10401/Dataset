"""Fixed Exp442 lifecycle wrappers with no path or scientific override surface."""
from __future__ import annotations

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    FrozenEvaluationAntigravity,
    FrozenPublicationAntigravity,
    ReadinessBindingAntigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity import (
    _run_readiness_for_profile_antigravity,
    _validate_readiness_for_profile_antigravity,
    current_clean_head_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import (
    _run_new_for_profile_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.reporting_antigravity import (
    _inspect_for_profile_antigravity,
    _publish_for_profile_antigravity,
)

from .profile_antigravity import PROFILE_ANTIGRAVITY


def run_readiness_antigravity() -> ReadinessBindingAntigravity:
    return _run_readiness_for_profile_antigravity(
        PROFILE_ANTIGRAVITY, current_clean_head_antigravity(),
    )


def run_new_antigravity() -> FrozenPublicationAntigravity:
    binding = _validate_readiness_for_profile_antigravity(
        PROFILE_ANTIGRAVITY, current_clean_head_antigravity(),
    )
    if current_clean_head_antigravity() != binding.reviewed_commit:
        raise ValueError("HEAD changed after Exp442 readiness validation")
    attempt = _run_new_for_profile_antigravity(
        PROFILE_ANTIGRAVITY, reviewed_commit=binding.reviewed_commit,
    )
    return _publish_for_profile_antigravity(PROFILE_ANTIGRAVITY, attempt)


def inspect_antigravity() -> FrozenPublicationAntigravity:
    return _inspect_for_profile_antigravity(PROFILE_ANTIGRAVITY)


def evaluate_antigravity() -> FrozenEvaluationAntigravity:
    from experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity import (
        _evaluate_for_profile_antigravity,
    )
    return _evaluate_for_profile_antigravity(PROFILE_ANTIGRAVITY)


__all__ = [
    "run_readiness_antigravity", "run_new_antigravity", "inspect_antigravity",
    "evaluate_antigravity",
]
