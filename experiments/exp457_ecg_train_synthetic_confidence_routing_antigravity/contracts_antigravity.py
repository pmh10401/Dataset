"""Immutable validated result contracts for Exp457 TRAIN evidence and routes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

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


__all__ = (
    "RouteDecisionAntigravity",
    "TrainQualificationAntigravity",
    "TrainSeedEvidenceAntigravity",
)
