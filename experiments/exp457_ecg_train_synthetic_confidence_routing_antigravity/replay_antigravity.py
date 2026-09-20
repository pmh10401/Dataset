"""Semantic replay for the exact nine-leaf Exp457 result artifact."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    DatasetRoutingFeaturesAntigravity,
)

from .artifacts_antigravity import (
    _LEAVES_ANTIGRAVITY,
    _payloads_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    EvaluationResultAntigravity,
    Exp456ResultAuthorityAntigravity,
    PublicationEvidenceAntigravity,
    RouteDecisionAntigravity,
    TrainQualificationAntigravity,
    TrainSeedEvidenceAntigravity,
)
from .evaluation_antigravity import (
    capture_exp456_result_authority_antigravity,
    evaluate_routes_antigravity,
)
from .router_antigravity import route_ecg_antigravity


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate artifact JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=no_duplicates_antigravity
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact JSON is invalid") from error
    canonical = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )
    if payload != canonical:
        raise ValueError("artifact JSON is not canonical")
    return value


def _strict_jsonl_antigravity(payload: bytes) -> tuple[object, ...]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("artifact JSONL is invalid")
    return tuple(
        _strict_json_antigravity(line + b"\n") for line in payload.splitlines()
    )


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
    artifact_map = _strict_json_antigravity(
        leaves["artifact_sha256_antigravity.json"]
    )
    expected = {
        name: sha256(leaves[name]).hexdigest() for name in _LEAVES_ANTIGRAVITY[:-1]
    }
    if artifact_map != expected:
        raise ValueError("artifact hash map differs")
    return leaves


def _qualifications_antigravity(payload: bytes) -> tuple[TrainQualificationAntigravity, ...]:
    value = _strict_json_antigravity(payload)
    if type(value) is not dict or set(value) != {"qualifications"} or type(
        value["qualifications"]
    ) is not list:
        raise ValueError("qualification artifact is invalid")
    qualifications = []
    qualification_keys = {"dataset", "seed_evidence", "passed"}
    seed_keys = {
        "dataset",
        "seed",
        "normal_trials",
        "normal_alerts",
        "normal_wilson_upper",
        "pooled_sensitivity",
        "per_type_sensitivity",
        "passed",
    }
    for row in value["qualifications"]:
        if type(row) is not dict or set(row) != qualification_keys or type(
            row["seed_evidence"]
        ) is not list:
            raise ValueError("qualification row is invalid")
        seeds = []
        for seed in row["seed_evidence"]:
            if type(seed) is not dict or set(seed) != seed_keys or type(
                seed["per_type_sensitivity"]
            ) is not dict or set(seed["per_type_sensitivity"]) != set(
                FROZEN_CONFIG_ANTIGRAVITY.primary_types
            ):
                raise ValueError("qualification seed row is invalid")
            sensitivities = {
                name: seed["per_type_sensitivity"][name]
                for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
            }
            seeds.append(
                TrainSeedEvidenceAntigravity(
                    seed["dataset"],
                    seed["seed"],
                    seed["normal_trials"],
                    seed["normal_alerts"],
                    seed["normal_wilson_upper"],
                    seed["pooled_sensitivity"],
                    sensitivities,
                    seed["passed"],
                )
            )
        qualifications.append(
            TrainQualificationAntigravity(
                row["dataset"], tuple(seeds), row["passed"]
            )
        )
    return tuple(qualifications)


def _features_antigravity(payload: bytes) -> tuple[DatasetRoutingFeaturesAntigravity, ...]:
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
    result = []
    for row in _strict_jsonl_antigravity(payload):
        if type(row) is not dict or set(row) != keys:
            raise ValueError("feature artifact row is invalid")
        result.append(DatasetRoutingFeaturesAntigravity(**row))
    return tuple(result)


def _decisions_antigravity(payload: bytes) -> tuple[RouteDecisionAntigravity, ...]:
    keys = {"dataset", "train_gate", "structure_gate", "route"}
    result = []
    for row in _strict_jsonl_antigravity(payload):
        if type(row) is not dict or set(row) != keys:
            raise ValueError("route decision artifact row is invalid")
        result.append(RouteDecisionAntigravity(**row))
    return tuple(result)


def _inspect_result_with_authority_antigravity(
    root: Path, expected_authority: Exp456ResultAuthorityAntigravity
) -> EvaluationResultAntigravity:
    """Injected replay seam used by synthetic tests and the public capture path."""
    leaves = _load_antigravity(root)
    _strict_json_antigravity(leaves["context_antigravity.json"])
    qualifications = _qualifications_antigravity(
        leaves["train_qualification_antigravity.json"]
    )
    features = _features_antigravity(leaves["features_antigravity.jsonl"])
    decisions = _decisions_antigravity(
        leaves["route_decisions_antigravity.jsonl"]
    )
    authority_features = {
        feature.dataset: feature for feature in expected_authority.features
    }
    recomputed_decisions = tuple(
        route_ecg_antigravity(authority_features[item.dataset], item)
        for item in qualifications
    )
    if decisions != recomputed_decisions:
        raise ValueError("stored routes differ from exact Gate 1/router replay")
    result = evaluate_routes_antigravity(expected_authority, recomputed_decisions)
    evidence = PublicationEvidenceAntigravity(
        expected_authority, qualifications, features, result
    )
    expected_payloads = _payloads_antigravity(evidence)
    if any(
        leaves[name] != expected_payloads[name] for name in _LEAVES_ANTIGRAVITY[:-1]
    ):
        raise ValueError("stored result evidence differs from semantic replay")
    return result


def inspect_result_at_antigravity(root: Path) -> EvaluationResultAntigravity:
    """Capture Exp456 once, then semantically replay the Exp457 result."""
    authority = capture_exp456_result_authority_antigravity()
    return _inspect_result_with_authority_antigravity(root, authority)


__all__ = ("inspect_result_at_antigravity",)
