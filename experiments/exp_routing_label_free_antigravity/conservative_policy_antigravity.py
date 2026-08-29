"""Official feature-only adapter for the qualified Exp476 fallback policy."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from experiments.exp474_conservative_fallback_routing_antigravity.policy_antigravity import (
    FEATURE_KEYS_ANTIGRAVITY,
)
from experiments.exp476_minimum_zf_fallback_routing_antigravity.policy_antigravity import (
    MinimumZFGuardKNNPolicyAntigravity,
)

from .routing_rules_antigravity import route_dataset_antigravity


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
ARTIFACT_PATH_ANTIGRAVITY = Path(__file__).with_name(
    "official_exp476_policy_artifact_antigravity.json"
)
EXP476_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp476_minimum_zf_fallback_routing_antigravity/results_antigravity.json"
)
FEATURE_CACHE_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp474_conservative_fallback_routing_antigravity/train_features_antigravity.json"
)
ROUTING_RULES_PATH_ANTIGRAVITY = Path(__file__).with_name("routing_rules_antigravity.py")
EXPECTED_EXP476_SHA256_ANTIGRAVITY = (
    "8b6f881d25d0d2865e05c5a5b9ca204f78641e1364bb60d536e7dcc8ec21733f"
)
EXPECTED_ARTIFACT_SHA256_ANTIGRAVITY = (
    "b45f4124354d6bd612657b801634c503fa84b3585776cfde4a8f76d19b805545"
)
EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY = (
    "10cf37c2eca6336eacf25badb85653e75232dfaf7b6caeef66abb149241d20ba"
)
EXPECTED_ROUTING_RULES_SHA256_ANTIGRAVITY = (
    "fb4fa5e6088a23ad2135b695715d964f4428b37391ccbf3cd59f9d573fc5c8f2"
)
EXPECTED_POLICY_CONFIG_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
    "min_lower_zf_gain": 0.0,
}


@lru_cache(maxsize=1)
def load_official_policy_artifact_antigravity() -> dict[str, object]:
    """Load and validate the anonymous frozen policy artifact once."""
    if sha256(EXP476_RESULT_PATH_ANTIGRAVITY.read_bytes()).hexdigest() != EXPECTED_EXP476_SHA256_ANTIGRAVITY:
        raise RuntimeError("Exp476 result hash does not match the qualified official policy")
    artifact_bytes = ARTIFACT_PATH_ANTIGRAVITY.read_bytes()
    if sha256(artifact_bytes).hexdigest() != EXPECTED_ARTIFACT_SHA256_ANTIGRAVITY:
        raise RuntimeError("official policy artifact hash does not match the qualified artifact")
    if sha256(FEATURE_CACHE_PATH_ANTIGRAVITY.read_bytes()).hexdigest() != EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY:
        raise RuntimeError("official policy feature source hash changed")
    if sha256(ROUTING_RULES_PATH_ANTIGRAVITY.read_bytes()).hexdigest() != EXPECTED_ROUTING_RULES_SHA256_ANTIGRAVITY:
        raise RuntimeError("official proposal rules changed without requalification")
    artifact = json.loads(artifact_bytes)
    if artifact.get("schema") != "official_exp476_minimum_zf_policy_artifact_v1":
        raise ValueError("unsupported official policy artifact schema")
    if artifact.get("source_exp476_result_sha256") != EXPECTED_EXP476_SHA256_ANTIGRAVITY:
        raise ValueError("official artifact references a different Exp476 result")
    if artifact.get("source_feature_cache_sha256") != EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY:
        raise ValueError("official artifact references a different feature source")
    if artifact.get("source_routing_rules_sha256") != EXPECTED_ROUTING_RULES_SHA256_ANTIGRAVITY:
        raise ValueError("official artifact references different proposal rules")
    if artifact.get("proposal_alignment_change_count") != 1:
        raise ValueError("official artifact proposal alignment evidence changed")
    rows = artifact.get("training_rows")
    if artifact.get("row_count") != 962 or not isinstance(rows, list) or len(rows) != 962:
        raise ValueError("official artifact must contain exactly 962 training rows")
    if artifact.get("feature_keys") != list(FEATURE_KEYS_ANTIGRAVITY):
        raise ValueError("official artifact feature schema does not match Exp476")
    if artifact.get("policy_config") != EXPECTED_POLICY_CONFIG_ANTIGRAVITY:
        raise ValueError("official artifact policy configuration does not match Exp476")
    if any(not isinstance(row, dict) or "dataset" in row for row in rows):
        raise ValueError("official artifact training rows must not contain dataset identifiers")
    return artifact


@lru_cache(maxsize=1)
def _official_policy_antigravity() -> MinimumZFGuardKNNPolicyAntigravity:
    artifact = load_official_policy_artifact_antigravity()
    return MinimumZFGuardKNNPolicyAntigravity(
        **EXPECTED_POLICY_CONFIG_ANTIGRAVITY
    ).fit(artifact["training_rows"])  # type: ignore[arg-type]


def route_features_official_antigravity(features: Mapping[str, float]) -> str:
    """Return the final method using only seven numeric TRAIN features."""
    proposal = route_dataset_antigravity(features)
    return _official_policy_antigravity().route(features, proposal).method


__all__ = [
    "ARTIFACT_PATH_ANTIGRAVITY",
    "EXPECTED_ARTIFACT_SHA256_ANTIGRAVITY",
    "load_official_policy_artifact_antigravity",
    "route_features_official_antigravity",
]
