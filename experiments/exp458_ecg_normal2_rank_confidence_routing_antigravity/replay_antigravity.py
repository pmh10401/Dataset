"""Semantic replay for the exact nine-leaf Exp458 Task 4 result artifact."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    DatasetRoutingFeaturesAntigravity,
)

from .artifacts_antigravity import _LEAVES_ANTIGRAVITY, _payloads_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    EvaluationResultAntigravity,
    Exp456ResultAuthorityAntigravity,
    PublicationEvidenceAntigravity,
    RankQualificationAntigravity,
    RankSeedEvidenceAntigravity,
    RouteDecisionAntigravity,
)
from .evaluation_antigravity import capture_exp456_result_authority_antigravity, evaluate_routes_antigravity
from .router_antigravity import route_normal2_antigravity


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
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if payload != canonical:
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


def _qualification_antigravity(payload: bytes) -> RankQualificationAntigravity:
    rows = _strict_jsonl_antigravity(payload)
    if len(rows) != 1:
        raise ValueError("qualification artifact row count differs")
    row = rows[0]
    qualification_keys = {"dataset", "seed_evidence", "passed"}
    seed_keys = {
        "dataset",
        "seed",
        "normal_trials",
        "normal_alerts",
        "normal_wilson_upper",
        "per_type_auc",
        "pooled_auc",
        "passed",
    }
    if type(row) is not dict or set(row) != qualification_keys or type(row["seed_evidence"]) is not list:
        raise ValueError("qualification row is invalid")
    seeds = []
    for seed in row["seed_evidence"]:
        if type(seed) is not dict or set(seed) != seed_keys or type(seed["per_type_auc"]) is not dict:
            raise ValueError("qualification seed row is invalid")
        aucs = {name: seed["per_type_auc"][name] for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types}
        seeds.append(
            RankSeedEvidenceAntigravity(
                seed["dataset"],
                seed["seed"],
                seed["normal_trials"],
                seed["normal_alerts"],
                seed["normal_wilson_upper"],
                aucs,
                seed["pooled_auc"],
                seed["passed"],
            )
        )
    return RankQualificationAntigravity(row["dataset"], tuple(seeds), row["passed"])


def _feature_antigravity(payload: bytes) -> DatasetRoutingFeaturesAntigravity:
    rows = _strict_jsonl_antigravity(payload)
    if len(rows) != 1:
        raise ValueError("feature artifact row count differs")
    keys = {
        "dataset",
        "family",
        "target_length",
        "baseline_count",
        "candidate_count",
        "overlap_count",
        "union_count",
        "added_count",
        "removed_count",
        "overlap_ppm",
    }
    row = rows[0]
    if type(row) is not dict or set(row) != keys:
        raise ValueError("feature artifact row is invalid")
    return DatasetRoutingFeaturesAntigravity(**row)


def _decision_antigravity(payload: bytes) -> RouteDecisionAntigravity:
    rows = _strict_jsonl_antigravity(payload)
    if len(rows) != 1:
        raise ValueError("route artifact row count differs")
    row = rows[0]
    keys = {"dataset", "rank_gate", "structure_gate", "route"}
    if type(row) is not dict or set(row) != keys:
        raise ValueError("route decision artifact row is invalid")
    return RouteDecisionAntigravity(**row)


def _inspect_result_with_authority_antigravity(
    root: Path, expected_authority: Exp456ResultAuthorityAntigravity
) -> EvaluationResultAntigravity:
    leaves = _load_antigravity(root)
    _strict_json_antigravity(leaves["context_antigravity.json"])
    qualification = _qualification_antigravity(leaves["train_qualification_antigravity.jsonl"])
    feature = _feature_antigravity(leaves["features_antigravity.jsonl"])
    decision = _decision_antigravity(leaves["route_decisions_antigravity.jsonl"])
    authority_features = {item.dataset: item for item in expected_authority.features}
    if feature != authority_features[FROZEN_CONFIG_ANTIGRAVITY.dataset]:
        raise ValueError("stored feature differs from Exp456 authority")
    recomputed_decision = route_normal2_antigravity(feature, qualification)
    if decision != recomputed_decision:
        raise ValueError("stored route differs from exact rank/router replay")
    result = evaluate_routes_antigravity(expected_authority, recomputed_decision)
    evidence = PublicationEvidenceAntigravity(expected_authority, qualification, feature, result)
    expected_payloads = _payloads_antigravity(evidence)
    if any(leaves[name] != expected_payloads[name] for name in _LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("stored result evidence differs from semantic replay")
    return result


def inspect_result_at_antigravity(root: Path) -> EvaluationResultAntigravity:
    authority = capture_exp456_result_authority_antigravity()
    return _inspect_result_with_authority_antigravity(root, authority)


__all__ = ("inspect_result_at_antigravity",)
