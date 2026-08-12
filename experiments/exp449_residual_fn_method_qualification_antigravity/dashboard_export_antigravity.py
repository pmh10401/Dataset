"""Local-only, recursively sanitized Exp449 lifecycle status projection."""
from __future__ import annotations

from collections.abc import Mapping
import re


STATUS_FIELDS_ANTIGRAVITY = (
    "schema", "experiment_id", "phase", "status", "reviewed_commit",
    "target_family_count", "candidate_pair_count", "dataset_unit_total",
    "dataset_unit_completed", "method_family", "backend", "label_access_started",
)
_SENSITIVE_ANTIGRAVITY = frozenset({"dataset", "instance_index", "sha256", "path", "log", "database", "label", "labels", "raw_labels", "sql", "checkpoint", "secret", "audit"})
_RESULT_FIELDS_ANTIGRAVITY = ("schema", "experiment", "baseline", "candidates", "families", "status", "runtime")


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


def public_result_antigravity(result: Mapping[str, object]) -> Mapping[str, object]:
    """Project only aggregate Task-6 truth; reject all raw, identifier, and operational material."""
    if not isinstance(result, Mapping) or set(result) != set(_RESULT_FIELDS_ANTIGRAVITY) or not _safe_antigravity(result):
        raise ValueError("public result is invalid or sensitive")
    if result["schema"] != "exp449.public-method-qualification.v1" or result["experiment"] != "Exp449" or type(result["status"]) is not str:
        raise ValueError("public result identity differs")
    def counts(value: object) -> bool:
        return isinstance(value, Mapping) and set(value) == {"tp", "fp", "fn", "tn", "f1"} and all(type(value[key]) is int and value[key] >= 0 for key in ("tp", "fp", "fn", "tn")) and type(value["f1"]) is float
    allowed_families = tuple(("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves"))
    allowed_methods = {family: methods for family, methods in (("Crop", ("minirocket_local_gap", "multirocket_local_gap")), ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")), ("FordA", ("minirocket_local_gap", "hydra_local_gap")), ("FordB", ("multirocket_local_gap", "hydra_local_gap")), ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")), ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")), ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")), ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")))}
    if not counts(result["baseline"]) or type(result["families"]) is not list or tuple(result["families"]) != allowed_families:
        raise ValueError("public aggregate baseline differs")
    if type(result["candidates"]) is not list or any(not isinstance(item, Mapping) or set(item) != {"family", "method", "qualified", "global"} or item["family"] not in allowed_methods or item["method"] not in allowed_methods[item["family"]] or type(item["qualified"]) is not bool or not counts(item["global"]) for item in result["candidates"]):
        raise ValueError("public candidate aggregate differs")
    runtime = result["runtime"]
    if not isinstance(runtime, Mapping) or runtime != {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}:
        raise ValueError("public runtime aggregate differs")
    return {key: result[key] for key in _RESULT_FIELDS_ANTIGRAVITY}
