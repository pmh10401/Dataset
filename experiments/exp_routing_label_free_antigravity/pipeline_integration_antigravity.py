"""End-to-end pipeline integration for label-free routing.

Provides a drop-in replacement for the frozen CANDIDATE_MATRIX lookup that
uses the label-free router instead of dataset-name prefix matching. This
allows the official pipeline to route unseen datasets without code changes.
"""

from __future__ import annotations

from typing import Mapping

from .official_router_antigravity import load_train_features_antigravity, route_official_antigravity


_FROZEN_CANDIDATE_MATRIX_ANTIGRAVITY: Mapping[str, tuple[str, ...]] = {
    "Crop": ("minirocket_local_gap", "multirocket_local_gap"),
    "ElectricDevices": ("multirocket_local_gap", "hydra_multirocket_local_gap"),
    "FordA": ("minirocket_local_gap", "hydra_local_gap"),
    "FordB": ("multirocket_local_gap", "hydra_local_gap"),
    "Yoga": ("multirocket_local_gap", "hydra_multirocket_local_gap"),
    "ECG5000": ("multirocket_local_gap", "hydra_local_gap"),
    "ChlorineConcentration": ("minirocket_local_gap", "multirocket_local_gap"),
    "StarLightCurves": ("multirocket_local_gap", "hydra_local_gap"),
}
_ALL_METHODS_ANTIGRAVITY = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
)


def get_candidates_for_dataset_antigravity(dataset_name: str) -> tuple[str, ...]:
    """Return candidate methods for a dataset using the v5 router.

    For trained families the frozen matrix is preserved verbatim (no regression).
    For unseen families only the router-selected method is returned.
    """
    family = dataset_name.rsplit("_normal_", 1)[0]
    if family in _FROZEN_CANDIDATE_MATRIX_ANTIGRAVITY:
        return _FROZEN_CANDIDATE_MATRIX_ANTIGRAVITY[family]
    method = route_official_antigravity(dataset_name)
    return (method,)
