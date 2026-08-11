"""Closed, single-use Exp446 recomputation lifecycle facade."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import (
    parse_canonical_json_bytes_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import (
    capture_exp446_terminal_authority_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP446_PROFILE_ANTIGRAVITY as _EXP446_PROFILE_ANTIGRAVITY


def run_readiness_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import readiness_for_profile_antigravity
    return readiness_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


def run_new_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import run_new_for_profile_antigravity
    return run_new_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


def _require_terminal_parity_antigravity() -> None:
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    try:
        authority = capture_exp446_terminal_authority_antigravity(
            attempt,
            reviewed_design_path=_EXP446_PROFILE_ANTIGRAVITY.reviewed_design_path,
            preregistration_path=_EXP446_PROFILE_ANTIGRAVITY.preregistration_path,
            include_marker=True,
        )
        marker_bytes = authority.marker_bytes_antigravity
        payload = parse_canonical_json_bytes_antigravity(marker_bytes)
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Exp446 publication requires terminal parity")
    authorities = payload.get("authorities")
    terminal = payload.get("terminal_sha256")
    if (
        set(payload) != {"schema", "record_prefix_payload_sha256", "authorities", "terminal_sha256"}
        or payload.get("schema") != "exp446.terminal-parity.v1"
        or not isinstance(authorities, list) or len(authorities) != 2
        or any(type(authority) is not str or len(authority) != 64 for authority in authorities)
        or payload.get("record_prefix_payload_sha256") != authorities[0]
        or not isinstance(terminal, Mapping) or set(terminal) != set(authority.leaf_bytes)
    ):
        raise ValueError("Exp446 publication requires terminal parity")
    for name, expected in terminal.items():
        if type(expected) is not str or len(expected) != 64:
            raise ValueError("Exp446 publication requires terminal parity")
        if hashlib.sha256(authority.leaf_bytes[name]).hexdigest() != expected:
            raise ValueError("Exp446 publication requires terminal parity")
    aggregate = payload["record_prefix_payload_sha256"]
    if type(aggregate) is not str or len(aggregate) != 64:
        raise ValueError("Exp446 publication requires terminal parity")
    if aggregate != authority.record_prefix_payload_sha256:
        raise ValueError("Exp446 publication requires terminal parity")
    from experiments.exp445_exp444_publication_source_authority_repair_antigravity.predecessor_antigravity import verify_exp444_predecessor_antigravity
    from .predecessor_antigravity import verify_exp445_predecessor_antigravity
    fresh = (verify_exp444_predecessor_antigravity(), verify_exp445_predecessor_antigravity())
    state = parse_canonical_json_bytes_antigravity(authority.leaf_bytes["checkpoint_antigravity/state_antigravity.json"])
    if not isinstance(state, Mapping):
        raise ValueError("Exp446 publication requires terminal parity")
    prefix = tuple(state["record_sha256_prefix"])
    if (
        [item.record_prefix_payload_sha256 for item in fresh] != authorities
        or any(tuple(item.record_sha256_prefix) != prefix for item in fresh)
        or any(item.record_prefix_payload_sha256 != aggregate for item in fresh)
    ):
        raise ValueError("Exp446 predecessor changed before publication eligibility")


def publish_prediction_antigravity():
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    finalization_failure = attempt / "terminal_finalization_failure_antigravity.json"
    if finalization_failure.exists() or finalization_failure.is_symlink():
        raise ValueError("Exp446 terminal finalization failure evidence is present")
    _require_terminal_parity_antigravity()
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import _publish_exp446_prediction_antigravity
    return _publish_exp446_prediction_antigravity(_EXP446_PROFILE_ANTIGRAVITY)


def inspect_prediction_antigravity():
    from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import _inspect_exp446_publication_antigravity
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    return _inspect_exp446_publication_antigravity(attempt / "publication_antigravity")


__all__ = ["run_readiness_antigravity", "run_new_antigravity", "publish_prediction_antigravity", "inspect_prediction_antigravity"]
