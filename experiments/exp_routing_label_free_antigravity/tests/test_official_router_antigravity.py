"""Tests for official routing entry point."""

from __future__ import annotations

import sqlite3

import numpy as np
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


def test_wafer_routes_to_hydra_via_kurtosis_gate() -> None:
    method = route_official_antigravity("Wafer_normal_-1")
    assert method == "hydra_local_gap"


def test_missing_dataset_raises() -> None:
    with pytest.raises(ValueError):
        route_official_antigravity("NonexistentDataset_normal_99")
