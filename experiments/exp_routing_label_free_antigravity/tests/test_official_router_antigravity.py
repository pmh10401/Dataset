"""Tests for official routing entry point."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.exp_routing_label_free_antigravity.official_router_antigravity import (
    load_train_features_antigravity,
    route_official_antigravity,
)


def test_route_official_returns_valid_method() -> None:
    method = route_official_antigravity("Crop_normal_1")
    assert method in ("minirocket_local_gap", "multirocket_local_gap", "hydra_local_gap", "hydra_multirocket_local_gap")


def test_router_is_deterministic() -> None:
    a = route_official_antigravity("ECG5000_normal_2")
    b = route_official_antigravity("ECG5000_normal_2")
    assert a == b


@pytest.mark.parametrize("dataset", ["Crop_normal_1", "AllGestureWiimoteY_normal_3"])
def test_loader_matches_frozen_all_train_row_features_for_fixed_and_ragged_data(
    dataset: str,
) -> None:
    cache_path = (
        Path(__file__).resolve().parents[2]
        / "exp474_conservative_fallback_routing_antigravity"
        / "train_features_antigravity.json"
    )
    expected = json.loads(cache_path.read_text(encoding="utf-8"))[dataset]
    actual = load_train_features_antigravity(dataset)
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_wafer_returns_a_valid_exp476_policy_decision() -> None:
    method = route_official_antigravity("Wafer_normal_-1")
    assert method in (
        "minirocket_local_gap",
        "multirocket_local_gap",
        "hydra_local_gap",
        "hydra_multirocket_local_gap",
    )


def test_loader_excludes_only_nonfinite_rows_when_clean_rows_remain() -> None:
    features = load_train_features_antigravity("DodgerLoopDay_normal_1")
    assert features["n_train"] == 18.0
    assert all(math.isfinite(value) for value in features.values())
    assert route_official_antigravity("DodgerLoopDay_normal_1") in (
        "minirocket_local_gap",
        "multirocket_local_gap",
        "hydra_local_gap",
        "hydra_multirocket_local_gap",
    )


def test_missing_dataset_raises() -> None:
    with pytest.raises(ValueError):
        route_official_antigravity("NonexistentDataset_normal_99")
