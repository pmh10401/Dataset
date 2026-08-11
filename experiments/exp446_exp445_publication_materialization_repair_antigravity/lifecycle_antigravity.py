"""Closed, single-use Exp446 recomputation lifecycle facade."""
from __future__ import annotations

import json

from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP446_PROFILE_ANTIGRAVITY as _EXP446_PROFILE_ANTIGRAVITY


def run_readiness_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import readiness_for_profile_antigravity
    return readiness_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


def run_new_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import run_new_for_profile_antigravity
    return run_new_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


def _require_terminal_parity_antigravity() -> None:
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    marker = attempt / "terminal_parity_antigravity.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("Exp446 publication requires terminal parity")
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    authorities = payload.get("authorities")
    if (
        payload.get("schema") != "exp446.terminal-parity.v1"
        or not isinstance(authorities, list) or len(authorities) != 2
        or payload.get("record_prefix_payload_sha256") != authorities[0]
    ):
        raise ValueError("Exp446 publication requires terminal parity")
    from experiments.exp445_exp444_publication_source_authority_repair_antigravity.predecessor_antigravity import verify_exp444_predecessor_antigravity
    from .predecessor_antigravity import verify_exp445_predecessor_antigravity
    fresh = (verify_exp444_predecessor_antigravity(), verify_exp445_predecessor_antigravity())
    if [item.record_prefix_payload_sha256 for item in fresh] != authorities:
        raise ValueError("Exp446 predecessor changed before publication eligibility")


def publish_prediction_antigravity():
    _require_terminal_parity_antigravity()
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import _publish_exp446_prediction_antigravity
    return _publish_exp446_prediction_antigravity(_EXP446_PROFILE_ANTIGRAVITY)


def inspect_prediction_antigravity():
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import _inspect_exp446_publication_antigravity
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    return _inspect_exp446_publication_antigravity(attempt / "publication_antigravity")


__all__ = ["run_readiness_antigravity", "run_new_antigravity", "publish_prediction_antigravity", "inspect_prediction_antigravity"]
