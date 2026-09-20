"""Semantically replay a bounded Exp455 result artifact without TEST labels."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _context_from_authority_antigravity,
    _feature_rows_antigravity,
    _json_bytes_antigravity,
    _jsonl_bytes_antigravity,
    _payloads_antigravity,
)
from .authority_antigravity import capture_routing_authority_antigravity
from .contracts_antigravity import (
    ConfusionAntigravity,
    DatasetRoutingAuthorityAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FoldDecisionAntigravity,
    FrozenRoutingAuthorityAntigravity,
    RouteRuleAntigravity,
    RoutingEvaluationAntigravity,
)
from .features_antigravity import build_label_free_features_antigravity
from .evaluation_antigravity import evaluate_routing_antigravity
from .router_antigravity import leave_one_dataset_out_antigravity


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates_antigravity)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    if json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" != payload:
        raise ValueError("artifact JSON is not canonical")
    return value


def _jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload.endswith(b"\n") or not payload:
        raise ValueError("artifact JSONL is invalid")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _confusion_antigravity(value: object) -> ConfusionAntigravity:
    if type(value) is not dict or set(value) != {"tp", "fp", "fn", "tn"}:
        raise ValueError("artifact confusion is invalid")
    return ConfusionAntigravity(value["tp"], value["fp"], value["fn"], value["tn"])


def _load_antigravity(root: Path) -> dict[str, bytes]:
    if not root.is_dir() or root.is_symlink() or {entry.name for entry in root.iterdir()} != set(_LEAVES_ANTIGRAVITY):
        raise ValueError("artifact inventory differs")
    leaves = {}
    for name in _LEAVES_ANTIGRAVITY:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("artifact leaf is invalid")
        leaves[name] = path.read_bytes()
    artifact_map = _strict_json_antigravity(leaves[_LEAVES_ANTIGRAVITY[-1]])
    expected = {name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]}
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _reconstruct_antigravity(leaves: dict[str, bytes]) -> tuple[FrozenRoutingAuthorityAntigravity, tuple[DatasetRoutingFeaturesAntigravity, ...], tuple[FoldDecisionAntigravity, ...]]:
    context = _strict_json_antigravity(leaves["context_antigravity.json"])
    if type(context) is not dict or set(context) != {"experiment_id", "exp454_leaf_sha256", "rows"} or context["experiment_id"] != "Exp455" or type(context["rows"]) is not list or type(context["exp454_leaf_sha256"]) is not list:
        raise ValueError("context is invalid")
    rows = []
    for row in context["rows"]:
        if type(row) is not dict or set(row) != {"dataset", "family", "target_length", "method", "baseline_confusion", "candidate_confusion"}:
            raise ValueError("context row is invalid")
        candidate = None if row["candidate_confusion"] is None else _confusion_antigravity(row["candidate_confusion"])
        rows.append(DatasetRoutingAuthorityAntigravity(row["dataset"], row["family"], row["target_length"], row["method"], frozenset(), None if candidate is None else frozenset(), _confusion_antigravity(row["baseline_confusion"]), candidate))
    feature_rows = []
    feature_keys = {"dataset", "family", "target_length", "baseline_count", "candidate_count", "overlap_count", "union_count", "added_count", "removed_count", "overlap_ppm"}
    for row in _jsonl_antigravity(leaves["features_antigravity.jsonl"]):
        if type(row) is not dict or set(row) != feature_keys:
            raise ValueError("feature row is invalid")
        feature_rows.append(DatasetRoutingFeaturesAntigravity(**row))
    decisions = []
    decision_keys = {"dataset", "family", "method", "selected_route", "rule"}
    for row in _jsonl_antigravity(leaves["fold_decisions_antigravity.jsonl"]):
        if type(row) is not dict or set(row) != decision_keys or type(row["rule"]) is not dict:
            raise ValueError("decision row is invalid")
        decisions.append(FoldDecisionAntigravity(row["dataset"], row["family"], row["method"], row["selected_route"], RouteRuleAntigravity(**row["rule"])))
    inventory = tuple(tuple(item) for item in context["exp454_leaf_sha256"])
    return FrozenRoutingAuthorityAntigravity(tuple(rows), inventory, 0), tuple(feature_rows), tuple(decisions)


def _inspect_replay_with_dependencies_antigravity(
    root: str | Path,
    expected_authority: FrozenRoutingAuthorityAntigravity,
    expected_features: tuple[DatasetRoutingFeaturesAntigravity, ...],
) -> RoutingEvaluationAntigravity:
    """Private test seam: compare stored evidence to independently frozen Task 2 inputs."""
    leaves = _load_antigravity(Path(root))
    expected_context = _json_bytes_antigravity(
        _context_from_authority_antigravity(expected_authority.rows, expected_authority.exp454_leaf_sha256)
    )
    expected_feature_rows = _jsonl_bytes_antigravity(_feature_rows_antigravity(expected_features))
    if leaves["context_antigravity.json"] != expected_context or leaves["features_antigravity.jsonl"] != expected_feature_rows:
        raise ValueError("stored authority or label-free features differ from frozen Task 2 inputs")
    _stored_authority, _stored_features, decisions = _reconstruct_antigravity(leaves)
    outcomes = {
        row.dataset: (row.baseline_confusion, row.candidate_confusion)
        for row in expected_authority.rows if row.candidate_confusion is not None
    }
    expected_decisions = leave_one_dataset_out_antigravity(expected_features, outcomes)
    if expected_decisions != decisions:
        raise ValueError("stored fold decisions differ from semantic replay")
    evaluation = evaluate_routing_antigravity(expected_authority, expected_features, decisions)
    expected_payloads = _payloads_antigravity(evaluation)
    if any(leaves[name] != expected_payloads[name] for name in _LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("stored evaluation evidence differs from semantic replay")
    return evaluation


def inspect_replay_at_antigravity(root: str | Path) -> RoutingEvaluationAntigravity:
    """Capture frozen Task 2 evidence once, then replay against that immutable authority."""
    authority = capture_routing_authority_antigravity()
    features = build_label_free_features_antigravity(authority)
    return _inspect_replay_with_dependencies_antigravity(root, authority, features)


__all__ = ["inspect_replay_at_antigravity"]
