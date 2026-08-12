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
    result_fields = (*_RESULT_FIELDS_ANTIGRAVITY, "repeatable", "qualified_candidate_count", "routing_research_eligible")
    if not isinstance(result, Mapping) or set(result) != set(result_fields) or not _safe_antigravity(result):
        raise ValueError("public result is invalid or sensitive")
    if result["schema"] != "exp449.public-method-qualification.v1" or result["experiment"] != "Exp449" or type(result["status"]) is not str:
        raise ValueError("public result identity differs")
    def counts(value: object) -> bool:
        return isinstance(value, Mapping) and set(value) == {"tp", "fp", "fn", "tn", "f1"} and all(type(value[key]) is int and value[key] >= 0 for key in ("tp", "fp", "fn", "tn")) and type(value["f1"]) is float
    allowed_families = tuple(("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves"))
    allowed_methods = {family: methods for family, methods in (("Crop", ("minirocket_local_gap", "multirocket_local_gap")), ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")), ("FordA", ("minirocket_local_gap", "hydra_local_gap")), ("FordB", ("multirocket_local_gap", "hydra_local_gap")), ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")), ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")), ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")), ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")))}
    if not counts(result["baseline"]) or result["baseline"]["f1"] != (0.0 if 2 * result["baseline"]["tp"] + result["baseline"]["fp"] + result["baseline"]["fn"] == 0 else (2.0 * result["baseline"]["tp"]) / (2 * result["baseline"]["tp"] + result["baseline"]["fp"] + result["baseline"]["fn"])) or type(result["families"]) is not list or tuple(result["families"]) != allowed_families:
        raise ValueError("public aggregate baseline differs")
    expected_pairs = tuple((family, method) for family, methods in allowed_methods.items() for method in methods)
    def valid_candidate(item: object, pair: tuple[str, str]) -> bool:
        if not isinstance(item, Mapping) or set(item) != {"family", "method", "target_fn_delta", "gates", "qualified", "global"} or (item.get("family"), item.get("method")) != pair or type(item["target_fn_delta"]) is not int or type(item["qualified"]) is not bool or not counts(item["global"]) or not isinstance(item["gates"], Mapping) or set(item["gates"]) != {"global_fn", "global_fp", "global_f1", "target_fn_delta"} or any(type(value) is not bool for value in item["gates"].values()):
            return False
        global_ = item["global"]; denominator = 2 * global_["tp"] + global_["fp"] + global_["fn"]
        if global_["f1"] != (0.0 if denominator == 0 else (2.0 * global_["tp"]) / denominator): return False
        gates = {"global_fn": global_["fn"] < 1988, "global_fp": global_["fp"] <= 314, "global_f1": global_["f1"] >= 0.6264199935, "target_fn_delta": item["target_fn_delta"] < 0}
        return dict(item["gates"]) == gates and item["qualified"] is all(gates.values())
    if type(result["candidates"]) is not list or len(result["candidates"]) != 16 or any(not valid_candidate(item, pair) for item, pair in zip(result["candidates"], expected_pairs, strict=True)):
        raise ValueError("public candidate aggregate differs")
    runtime = result["runtime"]
    if not isinstance(runtime, Mapping) or runtime != {"label_sessions": 1, "label_queries": 48, "label_rows": 72300}:
        raise ValueError("public runtime aggregate differs")
    if result["status"] != "synthetic_complete" or type(result["repeatable"]) is not bool or result["repeatable"] is not True or type(result["qualified_candidate_count"]) is not int or result["qualified_candidate_count"] != sum(item["qualified"] for item in result["candidates"]) or type(result["routing_research_eligible"]) is not bool or result["routing_research_eligible"] is not (result["qualified_candidate_count"] > 0):
        raise ValueError("public qualification summary differs")
    return {key: result[key] for key in result_fields}
