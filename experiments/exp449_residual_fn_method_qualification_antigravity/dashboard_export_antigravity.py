"""Local-only, recursively sanitized Exp449 lifecycle status projection."""
from __future__ import annotations

from collections.abc import Mapping
import re


STATUS_FIELDS_ANTIGRAVITY = (
    "schema", "experiment_id", "phase", "status", "reviewed_commit",
    "target_family_count", "candidate_pair_count", "dataset_unit_total",
    "dataset_unit_completed", "method_family", "backend", "label_access_started",
)
_SENSITIVE_ANTIGRAVITY = frozenset({"dataset", "instance_index", "sha256", "path", "log", "database", "label", "labels", "sql"})


def _safe_antigravity(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(type(key) is str and key.lower() not in _SENSITIVE_ANTIGRAVITY and _safe_antigravity(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(_safe_antigravity(item) for item in value)
    return not (type(value) is str and any(token in value.lower() for token in _SENSITIVE_ANTIGRAVITY))


def public_status_antigravity(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    """Return only the fixed twelve non-sensitive fields; no publication occurs here."""
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(STATUS_FIELDS_ANTIGRAVITY):
        raise ValueError("status snapshot is invalid")
    if any(type(key) is not str and True for key in snapshot) or any(key.lower() in _SENSITIVE_ANTIGRAVITY for key in snapshot) or not _safe_antigravity(snapshot):
        raise ValueError("status contains sensitive data")
    if not set(STATUS_FIELDS_ANTIGRAVITY).issubset(snapshot):
        raise ValueError("status fields are incomplete")
    string_fields = ("schema", "experiment_id", "phase", "status", "reviewed_commit", "method_family", "backend")
    integer_fields = ("target_family_count", "candidate_pair_count", "dataset_unit_total", "dataset_unit_completed")
    if any(type(snapshot[field]) is not str or not snapshot[field] for field in string_fields) or any(type(snapshot[field]) is not int or snapshot[field] < 0 for field in integer_fields) or type(snapshot["label_access_started"]) is not bool or snapshot["schema"] != "exp449" or not re.fullmatch(r"[0-9a-f]{40}", snapshot["reviewed_commit"]):
        raise ValueError("status field types are invalid")
    return {key: snapshot[key] for key in STATUS_FIELDS_ANTIGRAVITY}


def snapshot_status_antigravity(**values: object) -> Mapping[str, object]:
    """Validate the controller's local aggregate before its owner-only projection."""
    return public_status_antigravity(values)
