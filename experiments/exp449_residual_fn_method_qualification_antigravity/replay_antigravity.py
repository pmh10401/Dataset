"""Source-free semantic replay for the exact Task-6 publication inventory."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import stat
from types import MappingProxyType

from .artifacts_antigravity import canonical_json_bytes_antigravity, strict_json_bytes_antigravity
from .bootstrap_antigravity import BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY
from .config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY, FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import CandidateMetricAntigravity, ConfusionAntigravity


_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "compatibility_antigravity.json", "candidate_registry_antigravity.json",
    "frozen_predictions_antigravity.jsonl", "prediction_manifest_antigravity.json", "evaluation_antigravity.json",
    "candidate_metrics_antigravity.jsonl", "family_metrics_antigravity.jsonl", "bootstrap_antigravity.json",
    "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json",
)


@dataclass(frozen=True, slots=True)
class FrozenEvaluationAntigravity:
    reviewed_commit: str
    candidate_count: int
    qualified_count: int
    label_session_count: int
    label_query_count: int
    label_row_count: int
    candidate_metrics: tuple[CandidateMetricAntigravity, ...]
    artifact_map_sha256: str
    decision_sha256: str
    routing_research_eligible: bool

    def __post_init__(self) -> None:
        if type(self.reviewed_commit) is not str or len(self.reviewed_commit) != 40 or any(ch not in "0123456789abcdef" for ch in self.reviewed_commit):
            raise ValueError("reviewed commit is invalid")
        if (type(self.candidate_count) is not int or type(self.qualified_count) is not int or type(self.label_session_count) is not int
                or type(self.label_query_count) is not int or type(self.label_row_count) is not int):
            raise ValueError("evaluation count type differs")
        if (self.candidate_count, self.label_session_count, self.label_query_count, self.label_row_count) != (16, 1, 48, 72300):
            raise ValueError("evaluation fixed population differs")
        if type(self.candidate_metrics) is not tuple or len(self.candidate_metrics) != 16 or any(type(item) is not CandidateMetricAntigravity for item in self.candidate_metrics):
            raise ValueError("candidate metrics differ")
        if self.qualified_count != sum(item.qualified for item in self.candidate_metrics) or type(self.routing_research_eligible) is not bool or self.routing_research_eligible:
            raise ValueError("qualification result differs")
        for value in (self.artifact_map_sha256, self.decision_sha256):
            if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError("evaluation hash differs")


def _identity_antigravity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _pin_leaves_antigravity(root: Path) -> dict[str, bytes]:
    if not isinstance(root, Path):
        raise ValueError("publication root is invalid")
    root_info = os.lstat(root)
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("publication root is unsafe")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        before = _identity_antigravity(os.fstat(root_fd))
        if _identity_antigravity(root_info) != before or set(os.listdir(root_fd)) != set(_LEAVES_ANTIGRAVITY):
            raise ValueError("publication inventory is not exact")
        leaves: dict[str, bytes] = {}
        identities: dict[str, tuple[int, int, int, int, int, int, int]] = {}
        for name in _LEAVES_ANTIGRAVITY:
            leaf_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
            try:
                info = os.fstat(leaf_fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("publication leaf is unsafe")
                identities[name] = _identity_antigravity(info)
                leaves[name] = b"".join(iter(lambda: os.read(leaf_fd, 1 << 20), b""))
                if _identity_antigravity(os.fstat(leaf_fd)) != identities[name]:
                    raise ValueError("publication leaf changed while pinned")
            finally:
                os.close(leaf_fd)
        if _identity_antigravity(os.fstat(root_fd)) != before or _identity_antigravity(os.lstat(root)) != before:
            raise ValueError("publication directory changed while pinned")
        for name, identity in identities.items():
            if _identity_antigravity(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != identity:
                raise ValueError("publication leaf changed after read")
        return leaves
    finally:
        os.close(root_fd)


def _json_antigravity(leaves: Mapping[str, bytes], name: str) -> object:
    return strict_json_bytes_antigravity(leaves[name])


def _jsonl_antigravity(leaves: Mapping[str, bytes], name: str) -> tuple[object, ...]:
    payload = leaves[name]
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("JSONL leaf is non-canonical")
    return tuple(strict_json_bytes_antigravity(line + b"\n") for line in payload.splitlines())


def _confusion_antigravity(value: object) -> ConfusionAntigravity:
    if not isinstance(value, dict) or set(value) != {"tp", "fp", "fn", "tn"}:
        raise ValueError("confusion schema differs")
    return ConfusionAntigravity(value["tp"], value["fp"], value["fn"], value["tn"])


def _f1_antigravity(confusion: ConfusionAntigravity) -> float:
    denominator = 2 * confusion.tp + confusion.fp + confusion.fn
    return 0.0 if denominator == 0 else (2.0 * confusion.tp) / denominator


def _qualifies_antigravity(confusion: ConfusionAntigravity, target_fn_delta: int) -> bool:
    if type(target_fn_delta) is not int:
        raise ValueError("target FN delta must be an exact integer")
    return (confusion.fn < 1988 and confusion.fp <= 314 and _f1_antigravity(confusion) >= FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"] and target_fn_delta < 0)


def _metric_antigravity(row: object) -> CandidateMetricAntigravity:
    expected = {"family", "method", "target_confusion", "global_confusion", "target_fn_delta", "global_f1", "qualified"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValueError("candidate metric schema differs")
    target, global_ = _confusion_antigravity(row["target_confusion"]), _confusion_antigravity(row["global_confusion"])
    if type(row["global_f1"]) is not float or row["global_f1"] != _f1_antigravity(global_):
        raise ValueError("candidate global F1 differs")
    qualified = _qualifies_antigravity(global_, row["target_fn_delta"])
    if type(row["qualified"]) is not bool or row["qualified"] is not qualified:
        raise ValueError("candidate qualification differs")
    return CandidateMetricAntigravity(row["family"], row["method"], target, global_, row["target_fn_delta"], row["global_f1"], qualified)


def inspect_replay_at_antigravity(root: Path) -> FrozenEvaluationAntigravity:
    """Reconstruct Task-6 authority from only twelve descriptor-pinned publication leaves."""
    leaves = _pin_leaves_antigravity(root)
    artifact_map = _json_antigravity(leaves, "artifact_sha256_antigravity.json")
    if not isinstance(artifact_map, dict) or set(artifact_map) != set(_LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("artifact hash map differs")
    for name in _LEAVES_ANTIGRAVITY[:-1]:
        if type(artifact_map[name]) is not str or artifact_map[name] != sha256(leaves[name]).hexdigest():
            raise ValueError("artifact leaf hash differs")
    context, compatibility, registry = (_json_antigravity(leaves, name) for name in _LEAVES_ANTIGRAVITY[:3])
    predictions = _jsonl_antigravity(leaves, "frozen_predictions_antigravity.jsonl")
    manifest, evaluation = (_json_antigravity(leaves, name) for name in ("prediction_manifest_antigravity.json", "evaluation_antigravity.json"))
    metric_rows, family_rows = _jsonl_antigravity(leaves, "candidate_metrics_antigravity.jsonl"), _jsonl_antigravity(leaves, "family_metrics_antigravity.jsonl")
    bootstrap, decision, dashboard = (_json_antigravity(leaves, name) for name in ("bootstrap_antigravity.json", "decision_antigravity.json", "dashboard_snapshot_antigravity.json"))
    expected_registry = [[family, list(methods)] for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY]
    if context != {"schema": "exp449", "reviewed_commit": evaluation.get("reviewed_commit") if isinstance(evaluation, dict) else None, "label_access_count": 1} or compatibility != {"schema": "exp449.task6", "predecessor": "exp448", "prediction_authority": "task5"} or registry != expected_registry:
        raise ValueError("publication context or compatibility differs")
    if not isinstance(manifest, dict) or manifest != {"schema": "exp449_prediction_manifest", "rows": 96, "candidate_pairs": 16, "frozen_prediction_sha256": sha256(leaves["frozen_predictions_antigravity.jsonl"]).hexdigest()} or len(predictions) != 96:
        raise ValueError("frozen prediction manifest differs")
    metrics = tuple(_metric_antigravity(row) for row in metric_rows)
    expected_order = tuple((family, method) for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY for method in methods)
    if tuple((item.family, item.method) for item in metrics) != expected_order:
        raise ValueError("candidate metric order differs")
    if not isinstance(evaluation, dict) or set(evaluation) != {"schema", "reviewed_commit", "candidate_count", "qualified_count", "label_session_count", "label_query_count", "label_row_count", "routing_research_eligible"}:
        raise ValueError("evaluation schema differs")
    if not isinstance(decision, dict) or decision != {"schema": "exp449.decision", "qualified_count": sum(item.qualified for item in metrics), "routing_research_eligible": False, "gates": {"global_fn_lt": 1988, "global_fp_lte": 314, "global_f1_nonregression": FROZEN_CONFIG_ANTIGRAVITY["global_f1_min"], "target_fn_delta_lt": 0}}:
        raise ValueError("decision semantics differ")
    if not isinstance(bootstrap, dict) or set(bootstrap) != {"draw_sha256", "comparisons"} or bootstrap["draw_sha256"] != BOOTSTRAP_DRAW_SHA256_ANTIGRAVITY or not isinstance(bootstrap["comparisons"], list) or len(bootstrap["comparisons"]) != 16:
        raise ValueError("bootstrap semantics differ")
    if not isinstance(family_rows, tuple) or len(family_rows) != 8 or not isinstance(dashboard, dict):
        raise ValueError("family or dashboard publication differs")
    expected_families = tuple(family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY)
    family_baselines: dict[str, ConfusionAntigravity] = {}
    for family, row in zip(expected_families, family_rows, strict=True):
        if not isinstance(row, dict) or set(row) != {"family", "candidate_count", "qualified_count", "baseline_target_confusion"} or row["family"] != family or row["candidate_count"] != 2 or type(row["qualified_count"]) is not int:
            raise ValueError("family metric schema differs")
        family_baselines[family] = _confusion_antigravity(row["baseline_target_confusion"])
        if row["qualified_count"] != sum(item.qualified for item in metrics if item.family == family):
            raise ValueError("family qualification differs")
    if set(dashboard) != {"schema", "experiment", "baseline", "candidates", "families", "status", "runtime"} or dashboard.get("schema") != "exp449.public-method-qualification.v1" or dashboard.get("experiment") != "Exp449" or dashboard.get("families") != list(expected_families) or type(dashboard.get("status")) is not str or not isinstance(dashboard["baseline"], dict):
        raise ValueError("dashboard publication schema differs")
    base_payload = dashboard["baseline"]
    if set(base_payload) != {"tp", "fp", "fn", "tn", "f1"} or type(base_payload["f1"]) is not float:
        raise ValueError("dashboard baseline differs")
    baseline_global = _confusion_antigravity({key: base_payload[key] for key in ("tp", "fp", "fn", "tn")})
    if base_payload["f1"] != _f1_antigravity(baseline_global):
        raise ValueError("dashboard baseline F1 differs")
    if not isinstance(dashboard["runtime"], dict) or dashboard["runtime"] != {"label_sessions": 1, "label_queries": 48, "label_rows": 72300} or not isinstance(dashboard["candidates"], list) or len(dashboard["candidates"]) != 16:
        raise ValueError("dashboard aggregate differs")
    expected_dashboard_rows = [{"family": item.family, "method": item.method, "qualified": item.qualified, "global": {**{"tp": item.global_confusion.tp, "fp": item.global_confusion.fp, "fn": item.global_confusion.fn, "tn": item.global_confusion.tn}, "f1": item.global_f1}} for item in metrics]
    if dashboard["candidates"] != expected_dashboard_rows:
        raise ValueError("dashboard candidate aggregate differs")
    for item in metrics:
        family_base = family_baselines[item.family]
        if (item.target_confusion.tp + item.target_confusion.fn, item.target_confusion.fp + item.target_confusion.tn) != (family_base.tp + family_base.fn, family_base.fp + family_base.tn):
            raise ValueError("candidate target confusion does not conserve its baseline")
        expected_global = ConfusionAntigravity(baseline_global.tp - family_base.tp + item.target_confusion.tp, baseline_global.fp - family_base.fp + item.target_confusion.fp, baseline_global.fn - family_base.fn + item.target_confusion.fn, baseline_global.tn - family_base.tn + item.target_confusion.tn)
        if item.global_confusion != expected_global or item.target_fn_delta != item.target_confusion.fn - family_base.fn:
            raise ValueError("candidate global confusion or target delta differs")
    result = FrozenEvaluationAntigravity(evaluation["reviewed_commit"], evaluation["candidate_count"], evaluation["qualified_count"], evaluation["label_session_count"], evaluation["label_query_count"], evaluation["label_row_count"], metrics, sha256(leaves["artifact_sha256_antigravity.json"]).hexdigest(), sha256(leaves["decision_antigravity.json"]).hexdigest(), evaluation["routing_research_eligible"])
    if result.qualified_count != sum(item.qualified for item in metrics):
        raise ValueError("evaluation qualified count differs")
    return result


def inspect_antigravity() -> FrozenEvaluationAntigravity:
    """Public inspection is deliberately unavailable until Task 7 supplies a reviewed root."""
    raise ValueError("Task-6 inspection requires the closed Task-7 reviewed publication root")


__all__ = ["FrozenEvaluationAntigravity", "inspect_antigravity", "inspect_replay_at_antigravity"]
