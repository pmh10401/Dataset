"""Immutable validated rank-confidence contracts for Exp458 Task 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

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
    if type(value) is not RankSeedEvidenceAntigravity:
        raise ValueError("rank seed evidence type is invalid")
    per_type_values = tuple(
        value.per_type_auc[name] for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
    )
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
    expected = all(seed.passed for seed in value.seed_evidence)
    if value.passed is not expected:
        raise ValueError("rank qualification differs from both-seed recomputation")
    return value


__all__ = (
    "RankQualificationAntigravity",
    "RankSeedEvidenceAntigravity",
    "validate_rank_qualification_semantics_antigravity",
    "validate_rank_seed_semantics_antigravity",
)
