"""End-to-end tests for pipeline integration."""

from __future__ import annotations

import pytest

from experiments.exp_routing_label_free_antigravity.pipeline_integration_antigravity import (
    _ALL_METHODS_ANTIGRAVITY,
    get_candidates_for_dataset_antigravity,
)
from experiments.exp_routing_label_free_antigravity.official_router_antigravity import route_official_antigravity


def test_trained_family_uses_frozen_matrix() -> None:
    candidates = get_candidates_for_dataset_antigravity("Crop_normal_1")
    assert candidates == ("minirocket_local_gap", "multirocket_local_gap")


def test_unseen_family_routes_to_single_method() -> None:
    candidates = get_candidates_for_dataset_antigravity("GunPoint_normal_1")
    assert len(candidates) == 1
    assert candidates[0] in _ALL_METHODS_ANTIGRAVITY


def test_router_consistency_with_direct_call() -> None:
    for ds in ("Wafer_normal_-1", "TwoLeadECG_normal_1", "FaceFour_normal_1"):
        via_pipeline = get_candidates_for_dataset_antigravity(ds)
        direct = route_official_antigravity(ds)
        assert len(via_pipeline) == 1 and via_pipeline[0] == direct


def test_invalid_dataset_raises() -> None:
    with pytest.raises(ValueError):
        get_candidates_for_dataset_antigravity("NonexistentDataset_normal_99")
