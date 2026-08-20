"""Source-free semantic replay for Exp460 six-leaf artifacts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _json_bytes_antigravity,
    _payloads_antigravity,
)
from .contracts_antigravity import (
    Exp459FallbackAuthorityAntigravity,
    FallbackEvaluationAntigravity,
    validate_exp459_fallback_authority_antigravity,
)
from .evaluation_antigravity import compare_fallback_rules_antigravity


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


def _strict_non_map_payloads_antigravity(leaves: dict[str, bytes]) -> None:
    _strict_json_antigravity(leaves["context_antigravity.json"])
    _strict_jsonl_antigravity(leaves["rule_comparisons_antigravity.jsonl"])
    _strict_jsonl_antigravity(leaves["failed_folds_antigravity.jsonl"])
    _strict_json_antigravity(leaves["decision_antigravity.json"])
    _strict_json_antigravity(leaves["dashboard_snapshot_antigravity.json"])


def inspect_replay_at_antigravity(
    root: Path,
    authority: Exp459FallbackAuthorityAntigravity,
) -> FallbackEvaluationAntigravity:
    trusted_authority = validate_exp459_fallback_authority_antigravity(authority)
    trusted_result = compare_fallback_rules_antigravity(trusted_authority)
    leaves = _load_antigravity(root)
    _strict_non_map_payloads_antigravity(leaves)
    expected_payloads = _payloads_antigravity(trusted_authority, trusted_result)
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if leaves[name] != expected_payloads[name]:
            raise ValueError("artifact payload differs from trusted fallback evaluation")
    return trusted_result


__all__ = ("inspect_replay_at_antigravity",)
