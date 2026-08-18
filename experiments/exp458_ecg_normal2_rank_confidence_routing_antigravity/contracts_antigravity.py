"""Immutable validated rank-confidence contracts for Exp458 Task 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType

from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
    ComparisonMetricAntigravity,
    DatasetRoutingFeaturesAntigravity,
    FrozenRoutingAuthorityAntigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


def _dataset_antigravity(value: object) -> str:
    if type(value) is not str or value != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        raise ValueError("dataset must be the exact normal_2 dataset")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _positive_int_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_int_antigravity(value: object, name: str, *, upper: int) -> int:
    if type(value) is not int or value < 0 or value > upper:
        raise ValueError(f"{name} must be an integer between zero and {upper}")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _validate_per_type_auc_antigravity(value: object) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ValueError("per_type_auc must be a mapping")
    if tuple(value) != FROZEN_CONFIG_ANTIGRAVITY.primary_types:
        raise ValueError("per_type_auc must contain the exact ordered primary types")
    frozen = {
        name: _unit_float_antigravity(value[name], f"per_type_auc[{name}]")
        for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    }
    return MappingProxyType(frozen)


def _validated_seed_fields_antigravity(value: object) -> tuple[float, ...]:
    if type(value) is not RankSeedEvidenceAntigravity:
        raise ValueError("rank seed evidence type is invalid")
    _dataset_antigravity(value.dataset)
    if type(value.seed) is not int or value.seed not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the two exact Hydra seeds")
    _positive_int_antigravity(value.normal_trials, "normal_trials")
    _bounded_int_antigravity(
        value.normal_alerts,
        "normal_alerts",
        upper=value.normal_trials,
    )
    _unit_float_antigravity(value.normal_wilson_upper, "normal_wilson_upper")
    if type(value.per_type_auc) is not MappingProxyType:
        raise ValueError("per_type_auc must remain an exact validated mapping")
    per_type_auc = _validate_per_type_auc_antigravity(value.per_type_auc)
    _unit_float_antigravity(value.pooled_auc, "pooled_auc")
    _bool_antigravity(value.passed, "passed")
    return tuple(per_type_auc[name] for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types)


@dataclass(frozen=True, slots=True)
class RankSeedEvidenceAntigravity:
    dataset: str
    seed: int
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    per_type_auc: Mapping[str, float]
    pooled_auc: float
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed must be one of the two exact Hydra seeds")
        _positive_int_antigravity(self.normal_trials, "normal_trials")
        _bounded_int_antigravity(
            self.normal_alerts,
            "normal_alerts",
            upper=self.normal_trials,
        )
        _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        object.__setattr__(
            self,
            "per_type_auc",
            _validate_per_type_auc_antigravity(self.per_type_auc),
        )
        _unit_float_antigravity(self.pooled_auc, "pooled_auc")
        _bool_antigravity(self.passed, "passed")
        validate_rank_seed_semantics_antigravity(self)


def validate_rank_seed_semantics_antigravity(
    value: object,
) -> RankSeedEvidenceAntigravity:
    per_type_values = _validated_seed_fields_antigravity(value)
    expected_pooled = float(sum(per_type_values) / len(per_type_values))
    expected_passed = bool(
        value.normal_wilson_upper <= FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max
        and expected_pooled >= FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min
        and all(
            auc >= FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min for auc in per_type_values
        )
    )
    if value.pooled_auc != expected_pooled or value.passed is not expected_passed:
        raise ValueError("rank seed evidence differs from semantic recomputation")
    return value


@dataclass(frozen=True, slots=True)
class RankQualificationAntigravity:
    dataset: str
    seed_evidence: tuple[RankSeedEvidenceAntigravity, ...]
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.seed_evidence) is not tuple or len(self.seed_evidence) != len(
            FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        ):
            raise ValueError("seed_evidence must be the exact two-record tuple")
        if any(type(item) is not RankSeedEvidenceAntigravity for item in self.seed_evidence):
            raise ValueError("seed_evidence must contain RankSeedEvidenceAntigravity records")
        if tuple(item.seed for item in self.seed_evidence) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed_evidence must use both Hydra seeds in frozen order")
        if any(item.dataset != self.dataset for item in self.seed_evidence):
            raise ValueError("seed_evidence datasets must match qualification dataset")
        _bool_antigravity(self.passed, "passed")
        validate_rank_qualification_semantics_antigravity(self)


def validate_rank_qualification_semantics_antigravity(
    value: object,
) -> RankQualificationAntigravity:
    if type(value) is not RankQualificationAntigravity:
        raise ValueError("rank qualification type is invalid")
    _dataset_antigravity(value.dataset)
    if type(value.seed_evidence) is not tuple or len(value.seed_evidence) != len(
        FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
    ):
        raise ValueError("seed_evidence must be the exact two-record tuple")
    if any(type(item) is not RankSeedEvidenceAntigravity for item in value.seed_evidence):
        raise ValueError("seed_evidence must contain RankSeedEvidenceAntigravity records")
    if tuple(item.seed for item in value.seed_evidence) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed_evidence must use both Hydra seeds in frozen order")
    if any(item.dataset != value.dataset for item in value.seed_evidence):
        raise ValueError("seed_evidence datasets must match qualification dataset")
    _bool_antigravity(value.passed, "passed")
    expected = all(
        validate_rank_seed_semantics_antigravity(seed).passed for seed in value.seed_evidence
    )
    if value.passed is not expected:
        raise ValueError("rank qualification differs from both-seed recomputation")
    return value


def _nonempty_str_antigravity(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class RouteDecisionAntigravity:
    dataset: str
    rank_gate: bool
    structure_gate: bool
    route: str

    def __post_init__(self) -> None:
        _nonempty_str_antigravity(self.dataset, "dataset")
        _bool_antigravity(self.rank_gate, "rank_gate")
        _bool_antigravity(self.structure_gate, "structure_gate")
        if type(self.route) is not str or self.route not in {"baseline", "candidate"}:
            raise ValueError("route must be baseline or candidate")


@dataclass(frozen=True, slots=True)
class CriterionOutcomeAntigravity:
    metric: str
    operator: str
    threshold: int | float
    actual: int | float
    passed: bool

    def __post_init__(self) -> None:
        expected = {
            "final_fn": ("<=", int),
            "final_fp": ("<=", int),
            "final_f1": (">", float),
        }
        if type(self.metric) is not str or self.metric not in expected:
            raise ValueError("criterion metric is invalid")
        operator, scalar = expected[self.metric]
        if self.operator != operator or type(self.threshold) is not scalar or type(self.actual) is not scalar:
            raise ValueError("criterion scalar contract is invalid")
        if scalar is int:
            if self.threshold < 0 or self.actual < 0:
                raise ValueError("criterion integer values must be non-negative")
        elif not isfinite(self.threshold) or not isfinite(self.actual):
            raise ValueError("criterion float values must be finite")
        _bool_antigravity(self.passed, "criterion passed")


@dataclass(frozen=True, slots=True)
class FamilyMetricAntigravity:
    family: str
    fn_delta: int
    fp_delta: int

    def __post_init__(self) -> None:
        if type(self.family) is not str or self.family not in {
            "ECG5000",
            "ChlorineConcentration",
        }:
            raise ValueError("family metric family is invalid")
        if type(self.fn_delta) is not int or type(self.fp_delta) is not int:
            raise ValueError("family deltas must be exact integers")


@dataclass(frozen=True, slots=True)
class Exp456ResultAuthorityAntigravity:
    authority: FrozenRoutingAuthorityAntigravity
    features: tuple[DatasetRoutingFeaturesAntigravity, ...]
    predecessor_comparisons: tuple[ComparisonMetricAntigravity, ...]
    artifact_map_sha256: str
    comparison_sha256: str
    decision_sha256: str
    label_query_count: int = 0

    def __post_init__(self) -> None:
        if type(self.authority) is not FrozenRoutingAuthorityAntigravity:
            raise ValueError("Exp456 authority type is invalid")
        if type(self.features) is not tuple or any(
            type(item) is not DatasetRoutingFeaturesAntigravity for item in self.features
        ):
            raise ValueError("Exp456 feature tuple is invalid")
        if type(self.predecessor_comparisons) is not tuple or len(self.predecessor_comparisons) != 3 or any(
            type(item) is not ComparisonMetricAntigravity for item in self.predecessor_comparisons
        ):
            raise ValueError("Exp456 predecessor comparisons are invalid")
        for digest in (
            self.artifact_map_sha256,
            self.comparison_sha256,
            self.decision_sha256,
        ):
            if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("Exp456 result identity is invalid")
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("Exp456 authority must have label query count zero")


@dataclass(frozen=True, slots=True)
class EvaluationResultAntigravity:
    comparisons: tuple[ComparisonMetricAntigravity, ...]
    family_metrics: tuple[FamilyMetricAntigravity, ...]
    decisions: tuple[RouteDecisionAntigravity, ...]
    criteria: tuple[CriterionOutcomeAntigravity, ...]
    passed: bool
    status: str

    def __post_init__(self) -> None:
        if type(self.comparisons) is not tuple or len(self.comparisons) != 4 or any(
            type(item) is not ComparisonMetricAntigravity for item in self.comparisons
        ):
            raise ValueError("evaluation comparisons are invalid")
        if type(self.family_metrics) is not tuple or any(
            type(item) is not FamilyMetricAntigravity for item in self.family_metrics
        ):
            raise ValueError("evaluation family metrics are invalid")
        if type(self.decisions) is not tuple or any(
            type(item) is not RouteDecisionAntigravity for item in self.decisions
        ):
            raise ValueError("evaluation decisions are invalid")
        if type(self.criteria) is not tuple or any(
            type(item) is not CriterionOutcomeAntigravity for item in self.criteria
        ):
            raise ValueError("evaluation criteria are invalid")
        _bool_antigravity(self.passed, "evaluation passed")
        if type(self.status) is not str or self.status not in {
            "routing_candidate_qualified_for_external_validation",
            "routing_candidate_not_qualified_for_external_validation",
        }:
            raise ValueError("evaluation status is invalid")


@dataclass(frozen=True, slots=True)
class PublicationEvidenceAntigravity:
    exp456_authority: Exp456ResultAuthorityAntigravity
    qualification: RankQualificationAntigravity
    feature: DatasetRoutingFeaturesAntigravity
    result: EvaluationResultAntigravity


@dataclass(frozen=True, slots=True)
class PublishedResultAntigravity:
    root: Path
    leaf_sha256: tuple[tuple[str, str], ...]
    artifact_map_sha256: str


__all__ = (
    "CriterionOutcomeAntigravity",
    "EvaluationResultAntigravity",
    "Exp456ResultAuthorityAntigravity",
    "FamilyMetricAntigravity",
    "PublicationEvidenceAntigravity",
    "PublishedResultAntigravity",
    "RankQualificationAntigravity",
    "RankSeedEvidenceAntigravity",
    "RouteDecisionAntigravity",
    "validate_rank_qualification_semantics_antigravity",
    "validate_rank_seed_semantics_antigravity",
)
