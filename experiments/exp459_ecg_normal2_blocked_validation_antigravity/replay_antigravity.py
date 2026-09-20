"""Source-free semantic replay for Exp459 seven-leaf artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _json_bytes_antigravity,
    _payloads_antigravity,
)
from .contracts_antigravity import ValidationDecisionAntigravity
from .evaluation_antigravity import validate_frozen_validation_authority_antigravity


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates_antigravity)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if payload != _json_bytes_antigravity(value):
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root is invalid")
    if {entry.name for entry in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves: dict[str, bytes] = {}
    for name in _LEAVES_ANTIGRAVITY:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact leaf is invalid")
        leaves[name] = path.read_bytes()
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"])
    expected = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _exact_str_antigravity(value: object, name: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact string")
    return value


def _exact_int_antigravity(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact integer")
    return value


def _exact_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float:
        raise ValueError(f"{name} must be an exact float")
    return value


def _exact_bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be an exact bool")
    return value


def _dict_keys_antigravity(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} keys differ")
    return value


def _strict_non_map_payloads_antigravity(leaves: dict[str, bytes]) -> None:
    _strict_json_antigravity(leaves["context_antigravity.json"])
    _strict_jsonl_antigravity(leaves["fold_plan_antigravity.jsonl"])
    _strict_jsonl_antigravity(leaves["fold_evidence_antigravity.jsonl"])
    _strict_jsonl_antigravity(leaves["seed_summary_antigravity.jsonl"])
    _strict_json_antigravity(leaves["decision_antigravity.json"])
    _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])


def inspect_replay_at_antigravity(
    root: Path,
    authority: object,
) -> ValidationDecisionAntigravity:
    trusted = validate_frozen_validation_authority_antigravity(authority)
    leaves = _load_antigravity(root)
    _strict_non_map_payloads_antigravity(leaves)
    expected_payloads = _payloads_antigravity(trusted)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from trusted validation authority")
    return trusted.decision


def inspect_result_at_antigravity(root: Path) -> ValidationDecisionAntigravity:
    raise ValueError("trusted frozen validation authority is required; use inspect_replay_at_antigravity(root, authority)")


__all__ = ("inspect_replay_at_antigravity", "inspect_result_at_antigravity")
