"""Immutable validated result contracts for Exp457 TRAIN evidence and routes."""

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
    if type(value) is not str or value not in FROZEN_CONFIG_ANTIGRAVITY.ecg_datasets:
        raise ValueError("dataset must be one of the five exact ECG5000 datasets")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class TrainSeedEvidenceAntigravity:
    dataset: str
    seed: int
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    pooled_sensitivity: float
    per_type_sensitivity: Mapping[str, float]
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed must be one of the two exact Hydra seeds")
        if type(self.normal_trials) is not int or self.normal_trials < 1:
            raise ValueError("normal_trials must be a positive integer")
        if (
            type(self.normal_alerts) is not int
            or self.normal_alerts < 0
            or self.normal_alerts > self.normal_trials
        ):
            raise ValueError("normal_alerts must be an integer between zero and normal_trials")
        _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        _unit_float_antigravity(self.pooled_sensitivity, "pooled_sensitivity")
        if not isinstance(self.per_type_sensitivity, Mapping):
            raise ValueError("per_type_sensitivity must be a mapping")
        if tuple(self.per_type_sensitivity) != FROZEN_CONFIG_ANTIGRAVITY.primary_types:
            raise ValueError("per_type_sensitivity must contain the exact ordered primary types")
        frozen = {
            name: _unit_float_antigravity(
                self.per_type_sensitivity[name], f"per_type_sensitivity[{name}]"
            )
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        }
        object.__setattr__(self, "per_type_sensitivity", MappingProxyType(frozen))
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class TrainQualificationAntigravity:
    dataset: str
    seed_evidence: tuple[TrainSeedEvidenceAntigravity, ...]
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.seed_evidence) is not tuple or len(self.seed_evidence) != len(
            FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        ):
            raise ValueError("seed_evidence must be the exact two-record tuple")
        if any(type(item) is not TrainSeedEvidenceAntigravity for item in self.seed_evidence):
            raise ValueError("seed_evidence must contain TrainSeedEvidenceAntigravity records")
        if tuple(item.seed for item in self.seed_evidence) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed_evidence must use both Hydra seeds in frozen order")
        if any(item.dataset != self.dataset for item in self.seed_evidence):
            raise ValueError("seed_evidence datasets must match qualification dataset")
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class RouteDecisionAntigravity:
    dataset: str
    train_gate: bool
    structure_gate: bool
    route: str

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        _bool_antigravity(self.train_gate, "train_gate")
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
            type(item) is not ComparisonMetricAntigravity
            for item in self.predecessor_comparisons
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
        if type(self.comparisons) is not tuple or any(
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
    qualifications: tuple[TrainQualificationAntigravity, ...]
    features: tuple[DatasetRoutingFeaturesAntigravity, ...]
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
    "RouteDecisionAntigravity",
    "TrainQualificationAntigravity",
    "TrainSeedEvidenceAntigravity",
)
