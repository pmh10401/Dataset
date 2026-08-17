"""Conservative two-gate ECG routing for Exp457."""

from __future__ import annotations

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    DatasetRoutingFeaturesAntigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    RouteDecisionAntigravity,
    TrainQualificationAntigravity,
)


def _feature_antigravity(value: object) -> DatasetRoutingFeaturesAntigravity:
    if type(value) is not DatasetRoutingFeaturesAntigravity:
        raise ValueError("feature must be an Exp456 dataset routing feature")
    if value.dataset not in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets or value.family != "ECG5000":
        raise ValueError("feature must identify one exact ECG5000 dataset")
    numeric = (
        value.target_length,
        value.baseline_count,
        value.candidate_count,
        value.overlap_count,
        value.union_count,
        value.added_count,
        value.removed_count,
        value.overlap_ppm,
    )
    if any(type(item) is not int or item < 0 for item in numeric):
        raise ValueError("feature counts must be exact non-negative integers")
    return value


def route_ecg_antigravity(
    feature: DatasetRoutingFeaturesAntigravity,
    qualification: TrainQualificationAntigravity,
) -> RouteDecisionAntigravity:
    """Recompute both frozen gates; equality remains baseline."""
    value = _feature_antigravity(feature)
    if type(qualification) is not TrainQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    if qualification.dataset != value.dataset:
        raise ValueError("qualification and feature dataset keys differ")
    train_gate = qualification.passed
    structure_gate = value.added_count > value.removed_count
    return RouteDecisionAntigravity(
        value.dataset,
        train_gate,
        structure_gate,
        "candidate" if train_gate and structure_gate else "baseline",
    )


__all__ = ("route_ecg_antigravity",)
