"""Fixed Exp445 lifecycle facade without override-capable operator inputs."""
from __future__ import annotations

from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP445_PROFILE_ANTIGRAVITY as _EXP445_PROFILE_ANTIGRAVITY
from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import readiness_for_profile_antigravity as _readiness_for_profile_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import run_new_for_profile_antigravity as _run_new_for_profile_antigravity


def run_readiness_antigravity(reviewed_commit: str):
    """Run the fixed Exp445 label-free readiness contract."""
    return _readiness_for_profile_antigravity(_EXP445_PROFILE_ANTIGRAVITY, reviewed_commit)


def run_new_antigravity(reviewed_commit: str):
    """Run the fixed Exp445 single-use replay with record parity gates."""
    return _run_new_for_profile_antigravity(_EXP445_PROFILE_ANTIGRAVITY, reviewed_commit)


__all__ = ["run_readiness_antigravity", "run_new_antigravity"]
