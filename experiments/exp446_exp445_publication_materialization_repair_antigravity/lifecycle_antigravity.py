"""Closed, single-use Exp446 recomputation lifecycle facade."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    parse_canonical_json_bytes_antigravity,
    read_pinned_regular_bytes_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    ProfileCheckpointContextAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP446_PROFILE_ANTIGRAVITY as _EXP446_PROFILE_ANTIGRAVITY


def run_readiness_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import readiness_for_profile_antigravity
    return readiness_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


def run_new_antigravity(reviewed_commit: str):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.runtime_antigravity import run_new_for_profile_antigravity
    return run_new_for_profile_antigravity(_EXP446_PROFILE_ANTIGRAVITY, reviewed_commit)


_TERMINAL_LEAVES_ANTIGRAVITY = (
    "checkpoint_antigravity/context_antigravity.json",
    "checkpoint_antigravity/state_antigravity.json",
    "progress_antigravity.json",
    "pid_antigravity.json",
    "events_antigravity.jsonl",
    "text_antigravity.log",
    *(f"checkpoint_antigravity/records_antigravity/{index:04d}_record_antigravity.json" for index in range(30)),
)


def _terminal_leaf_bytes_antigravity(attempt, name: str) -> bytes:
    if type(name) is not str or name not in _TERMINAL_LEAVES_ANTIGRAVITY:
        raise ValueError("Exp446 publication requires terminal parity")
    try:
        payload, _ = read_pinned_regular_bytes_antigravity(attempt / name)
    except (OSError, ValueError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    return payload


def _canonical_terminal_object_antigravity(payload: bytes) -> Mapping[str, object]:
    try:
        value = parse_canonical_json_bytes_antigravity(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    if not isinstance(value, Mapping):
        raise ValueError("Exp446 publication requires terminal parity")
    return value


def _require_complete_terminal_evidence_antigravity(
    attempt, aggregate: str,
) -> None:
    context_bytes = _terminal_leaf_bytes_antigravity(attempt, "checkpoint_antigravity/context_antigravity.json")
    context = _canonical_terminal_object_antigravity(context_bytes)
    try:
        context_value = ProfileCheckpointContextAntigravity.from_exp446_payload_antigravity(context)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    for path, expected in (
        (_EXP446_PROFILE_ANTIGRAVITY.reviewed_design_path, context_value.reviewed_design_sha256),
        (_EXP446_PROFILE_ANTIGRAVITY.preregistration_path, context_value.preregistration_sha256),
    ):
        try:
            document, _ = read_pinned_regular_bytes_antigravity(path)
        except (OSError, ValueError) as error:
            raise ValueError("Exp446 publication requires terminal parity") from error
        if hashlib.sha256(document).hexdigest() != expected:
            raise ValueError("Exp446 publication requires terminal parity")

    state = _canonical_terminal_object_antigravity(
        _terminal_leaf_bytes_antigravity(attempt, "checkpoint_antigravity/state_antigravity.json")
    )
    if (
        state.get("context_sha256") != hashlib.sha256(context_bytes).hexdigest()
        or state.get("status") != "complete"
        or state.get("completed_units") != 30
        or state.get("total_units") != 30
        or state.get("error_count") != 0
        or state.get("current_unit") is not None
        or state.get("current_view") is not None
        or state.get("error") is not None
        or not isinstance(state.get("record_sha256_prefix"), list)
        or len(state["record_sha256_prefix"]) != 30
        or state.get("payload_sha256") != aggregate
    ):
        raise ValueError("Exp446 publication requires terminal parity")
    prefix = state["record_sha256_prefix"]
    if any(type(value) is not str or len(value) != 64 for value in prefix):
        raise ValueError("Exp446 publication requires terminal parity")
    if hashlib.sha256(canonical_json_bytes_antigravity(prefix)).hexdigest() != aggregate:
        raise ValueError("Exp446 publication requires terminal parity")
    for index, expected in enumerate(prefix):
        record = _terminal_leaf_bytes_antigravity(
            attempt, f"checkpoint_antigravity/records_antigravity/{index:04d}_record_antigravity.json",
        )
        record_payload = _canonical_terminal_object_antigravity(record)
        if set(record_payload) != {"sequence", "result"} or record_payload.get("sequence") != index:
            raise ValueError("Exp446 publication requires terminal parity")
        if hashlib.sha256(record).hexdigest() != expected:
            raise ValueError("Exp446 publication requires terminal parity")

    progress = _canonical_terminal_object_antigravity(
        _terminal_leaf_bytes_antigravity(attempt, "progress_antigravity.json")
    )
    pid = _canonical_terminal_object_antigravity(_terminal_leaf_bytes_antigravity(attempt, "pid_antigravity.json"))
    if (
        progress.get("stage") != "complete" or progress.get("active") is not False
        or progress.get("completed") != 30 or progress.get("recomputed") != 30
        or progress.get("error") != 0 or progress.get("total") != 30
        or pid.get("active") is not False
        or pid.get("pid") != progress.get("pid")
        or pid.get("start_identity") != progress.get("start_identity")
    ):
        raise ValueError("Exp446 publication requires terminal parity")
    events = _terminal_leaf_bytes_antigravity(attempt, "events_antigravity.jsonl")
    try:
        rows = [parse_canonical_json_bytes_antigravity(line + b"\n") for line in events.splitlines()]
        text = _terminal_leaf_bytes_antigravity(attempt, "text_antigravity.log").decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("Exp446 publication requires terminal parity") from error
    if not rows or not isinstance(rows[-1], Mapping) or rows[-1].get("event") != "terminal" or rows[-1].get("stage") != "complete" or "종료 stage=complete" not in text:
        raise ValueError("Exp446 publication requires terminal parity")


def _require_terminal_parity_antigravity() -> None:
    attempt = _EXP446_PROFILE_ANTIGRAVITY.run_root / "attempts_antigravity" / "attempt_0000_antigravity"
    marker = attempt / "terminal_parity_antigravity.json"
    try:
        marker_bytes, _ = read_pinned_regular_bytes_antigravity(marker)
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
        or not isinstance(terminal, Mapping) or set(terminal) != set(_TERMINAL_LEAVES_ANTIGRAVITY)
    ):
        raise ValueError("Exp446 publication requires terminal parity")
    for name, expected in terminal.items():
        if type(expected) is not str or len(expected) != 64:
            raise ValueError("Exp446 publication requires terminal parity")
        if hashlib.sha256(_terminal_leaf_bytes_antigravity(attempt, name)).hexdigest() != expected:
            raise ValueError("Exp446 publication requires terminal parity")
    aggregate = payload["record_prefix_payload_sha256"]
    if type(aggregate) is not str or len(aggregate) != 64:
        raise ValueError("Exp446 publication requires terminal parity")
    _require_complete_terminal_evidence_antigravity(attempt, aggregate)
    from experiments.exp445_exp444_publication_source_authority_repair_antigravity.predecessor_antigravity import verify_exp444_predecessor_antigravity
    from .predecessor_antigravity import verify_exp445_predecessor_antigravity
    fresh = (verify_exp444_predecessor_antigravity(), verify_exp445_predecessor_antigravity())
    state = _canonical_terminal_object_antigravity(
        _terminal_leaf_bytes_antigravity(attempt, "checkpoint_antigravity/state_antigravity.json")
    )
    prefix = tuple(state["record_sha256_prefix"])
    if (
        [item.record_prefix_payload_sha256 for item in fresh] != authorities
        or any(tuple(item.record_sha256_prefix) != prefix for item in fresh)
        or any(item.record_prefix_payload_sha256 != aggregate for item in fresh)
    ):
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
