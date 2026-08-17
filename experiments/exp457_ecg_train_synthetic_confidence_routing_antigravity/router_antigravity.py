"""Conservative two-gate ECG routing for Exp457."""

from __future__ import annotations

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    DatasetRoutingFeaturesAntigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    RouteDecisionAntigravity,
    TrainQualificationAntigravity,
    validate_train_qualification_semantics_antigravity,
)


_EVIDENCE_BY_DECISION_ID_ANTIGRAVITY: dict[
    int,
    tuple[
        RouteDecisionAntigravity,
        DatasetRoutingFeaturesAntigravity,
        TrainQualificationAntigravity,
    ],
] = {}


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


def _route_decision_antigravity(
    feature: DatasetRoutingFeaturesAntigravity,
    qualification: TrainQualificationAntigravity,
) -> RouteDecisionAntigravity:
    value = _feature_antigravity(feature)
    if type(qualification) is not TrainQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    if qualification.dataset != value.dataset:
        raise ValueError("qualification and feature dataset keys differ")
    validate_train_qualification_semantics_antigravity(qualification)
    train_gate = qualification.passed
    structure_gate = value.added_count > value.removed_count
    return RouteDecisionAntigravity(
        value.dataset,
        train_gate,
        structure_gate,
        "candidate" if train_gate and structure_gate else "baseline",
    )


def _evidence_for_route_decision_antigravity(
    decision: RouteDecisionAntigravity,
) -> tuple[DatasetRoutingFeaturesAntigravity, TrainQualificationAntigravity]:
    evidence = _EVIDENCE_BY_DECISION_ID_ANTIGRAVITY.get(id(decision))
    if evidence is None or evidence[0] is not decision:
        raise ValueError("route decision lacks exact Gate 1 provenance")
    _stored, feature, qualification = evidence
    if _route_decision_antigravity(feature, qualification) != decision:
        raise ValueError("route decision differs from semantic recomputation")
    return feature, qualification


def route_ecg_antigravity(
    feature: DatasetRoutingFeaturesAntigravity,
    qualification: TrainQualificationAntigravity,
) -> RouteDecisionAntigravity:
    """Recompute both frozen gates; equality remains baseline."""
    decision = _route_decision_antigravity(feature, qualification)
    _EVIDENCE_BY_DECISION_ID_ANTIGRAVITY[id(decision)] = (
        decision,
        feature,
        qualification,
    )
    return decision


__all__ = ("route_ecg_antigravity",)
