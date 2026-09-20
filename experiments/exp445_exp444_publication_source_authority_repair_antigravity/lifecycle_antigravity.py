"""Fixed Exp445 lifecycle facade without override-capable operator inputs."""
from __future__ import annotations

from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP445_PROFILE_ANTIGRAVITY as _EXP445_PROFILE_ANTIGRAVITY


def run_readiness_antigravity(reviewed_commit: str):
    """Run the fixed Exp445 label-free readiness contract."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import readiness_for_profile_antigravity

    return readiness_for_profile_antigravity(_EXP445_PROFILE_ANTIGRAVITY, reviewed_commit)


def run_new_antigravity(reviewed_commit: str):
    """Run the fixed Exp445 single-use replay with record parity gates."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import run_new_for_profile_antigravity

    return run_new_for_profile_antigravity(_EXP445_PROFILE_ANTIGRAVITY, reviewed_commit)


def _publish_exp445_prediction_antigravity(profile: object):
    """Keep the private patch seam lazy so inspection never imports sources."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import (
        _publish_exp445_prediction_antigravity as publish,
    )

    return publish(profile)


def publish_prediction_antigravity():
    """Publish the fixed Exp445 replay through its closed no-retry boundary."""
    return _publish_exp445_prediction_antigravity(_EXP445_PROFILE_ANTIGRAVITY)


def inspect_prediction_antigravity():
    """Inspect only the fixed Exp445 publication, without opening live sources."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import _inspect_exp445_publication_antigravity

    attempt = _EXP445_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    return _inspect_exp445_publication_antigravity(attempt / "publication_antigravity")


__all__ = [
    "run_readiness_antigravity", "run_new_antigravity", "publish_prediction_antigravity",
    "inspect_prediction_antigravity",
]
