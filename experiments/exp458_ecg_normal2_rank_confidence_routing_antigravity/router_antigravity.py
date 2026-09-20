"""Normal_2-only rank-confidence router for Exp458 Task 4."""

from __future__ import annotations

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    DatasetRoutingFeaturesAntigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    RankQualificationAntigravity,
    RouteDecisionAntigravity,
    validate_rank_qualification_semantics_antigravity,
)


_EVIDENCE_BY_DECISION_ID_ANTIGRAVITY: dict[
    int,
    tuple[
        RouteDecisionAntigravity,
        DatasetRoutingFeaturesAntigravity,
        RankQualificationAntigravity,
    ],
] = {}


def _feature_antigravity(value: object) -> DatasetRoutingFeaturesAntigravity:
    if type(value) is not DatasetRoutingFeaturesAntigravity:
        raise ValueError("feature must be an Exp456 dataset routing feature")
    if type(value.dataset) is not str or not value.dataset or type(value.family) is not str:
        raise ValueError("feature identity is invalid")
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
    qualification: RankQualificationAntigravity,
) -> RouteDecisionAntigravity:
    value = _feature_antigravity(feature)
    if type(qualification) is not RankQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    validate_rank_qualification_semantics_antigravity(qualification)
    structure_gate = value.added_count > value.removed_count
    if value.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        return RouteDecisionAntigravity(value.dataset, False, structure_gate, "baseline")
    if qualification.dataset != value.dataset:
        raise ValueError("qualification and normal_2 feature dataset keys differ")
    rank_gate = qualification.passed
    return RouteDecisionAntigravity(
        value.dataset,
        rank_gate,
        structure_gate,
        "candidate" if rank_gate and structure_gate else "baseline",
    )


def _evidence_for_route_decision_antigravity(
    decision: RouteDecisionAntigravity,
) -> tuple[DatasetRoutingFeaturesAntigravity, RankQualificationAntigravity]:
    evidence = _EVIDENCE_BY_DECISION_ID_ANTIGRAVITY.get(id(decision))
    if evidence is None or evidence[0] is not decision:
        raise ValueError("route decision lacks exact rank provenance")
    _stored, feature, qualification = evidence
    if _route_decision_antigravity(feature, qualification) != decision:
        raise ValueError("route decision differs from semantic recomputation")
    return feature, qualification


def route_normal2_antigravity(
    feature: DatasetRoutingFeaturesAntigravity,
    qualification: RankQualificationAntigravity,
) -> RouteDecisionAntigravity:
    """Route only ECG5000_normal_2; every other feature is forced baseline."""
    decision = _route_decision_antigravity(feature, qualification)
    _EVIDENCE_BY_DECISION_ID_ANTIGRAVITY[id(decision)] = (
        decision,
        feature,
        qualification,
    )
    return decision


__all__ = ("route_normal2_antigravity",)
