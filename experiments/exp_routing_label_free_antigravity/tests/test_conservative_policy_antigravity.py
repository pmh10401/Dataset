"""Tests for the officially promoted Exp476 conservative policy."""

from __future__ import annotations

from inspect import signature
from hashlib import sha256
import json

from experiments.exp476_minimum_zf_fallback_routing_antigravity.policy_antigravity import (
    MinimumZFGuardKNNPolicyAntigravity,
)
from experiments.exp_routing_label_free_antigravity.conservative_policy_antigravity import (
    ARTIFACT_PATH_ANTIGRAVITY,
    EXPECTED_ARTIFACT_SHA256_ANTIGRAVITY,
    load_official_policy_artifact_antigravity,
    route_features_official_antigravity,
)
from experiments.exp_routing_label_free_antigravity.routing_rules_antigravity import (
    route_dataset_antigravity,
)


EXPECTED_EXP476_SHA256_ANTIGRAVITY = (
    "8b6f881d25d0d2865e05c5a5b9ca204f78641e1364bb60d536e7dcc8ec21733f"
)


def test_artifact_is_complete_and_contains_no_dataset_identifiers() -> None:
    artifact_bytes = ARTIFACT_PATH_ANTIGRAVITY.read_bytes()
    raw = artifact_bytes.decode("utf-8")
    artifact = json.loads(raw)

    assert sha256(artifact_bytes).hexdigest() == EXPECTED_ARTIFACT_SHA256_ANTIGRAVITY
    assert artifact["source_exp476_result_sha256"] == EXPECTED_EXP476_SHA256_ANTIGRAVITY
    assert artifact["row_count"] == 962
    assert len(artifact["training_rows"]) == 962
    assert "_normal_" not in raw
    assert all("dataset" not in row for row in artifact["training_rows"])


def test_all_frozen_proposals_match_the_official_feature_rules() -> None:
    rows = load_official_policy_artifact_antigravity()["training_rows"]
    assert all(
        row["proposal"] == route_dataset_antigravity(row["features"])
        for row in rows
    )


def test_feature_router_has_no_dataset_name_parameter() -> None:
    assert tuple(signature(route_features_official_antigravity).parameters) == ("features",)


def test_official_adapter_matches_the_qualified_exp476_policy() -> None:
    artifact = load_official_policy_artifact_antigravity()
    rows = artifact["training_rows"]
    config = artifact["policy_config"]
    direct = MinimumZFGuardKNNPolicyAntigravity(**config).fit(rows)

    for row in rows[::97]:
        features = row["features"]
        proposal = route_dataset_antigravity(features)
        assert route_features_official_antigravity(features) == direct.route(features, proposal).method


def test_feature_router_is_deterministic() -> None:
    row = load_official_policy_artifact_antigravity()["training_rows"][17]
    features = row["features"]
    assert route_features_official_antigravity(features) == route_features_official_antigravity(features)
