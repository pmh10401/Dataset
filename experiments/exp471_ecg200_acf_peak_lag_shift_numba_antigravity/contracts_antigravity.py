"""Immutable Exp471 fold and qualification contracts; no scoring or routing lives here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
import re
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


def _count_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _mapping_antigravity(
    value: object,
    name: str,
    expected_keys: tuple[object, ...],
    validator: object,
) -> MappingProxyType:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(value) != expected_keys:
        raise ValueError(f"{name} must preserve the frozen key order")
    frozen: dict[object, object] = {}
    for key in expected_keys:
        frozen[key] = validator(value[key], f"{name}[{key!r}]")  # type: ignore[operator]
    return MappingProxyType(frozen)


def _auc_mapping_antigravity(value: object, name: str) -> MappingProxyType:
    return _mapping_antigravity(
        value,
        name,
        FROZEN_CONFIG_ANTIGRAVITY.change_types,
        _unit_float_antigravity,
    )


@dataclass(frozen=True, slots=True)
class PairRepairCountsAntigravity:
    fixed: int
    broken: int

    def __post_init__(self) -> None:
        _count_antigravity(self.fixed, "fixed")
        _count_antigravity(self.broken, "broken")


def _pair_repair_antigravity(value: object, name: str) -> PairRepairCountsAntigravity:
    if type(value) is not PairRepairCountsAntigravity:
        raise ValueError(f"{name} must be an exact PairRepairCountsAntigravity")
    return value


def _pair_repairs_mapping_antigravity(value: object) -> MappingProxyType:
    return _mapping_antigravity(
        value,
        "pair_repairs",
        FROZEN_CONFIG_ANTIGRAVITY.change_types,
        _pair_repair_antigravity,
    )


def _seed_count_mapping_antigravity(value: object, name: str) -> MappingProxyType:
    return _mapping_antigravity(
        value,
        name,
        FROZEN_CONFIG_ANTIGRAVITY.seeds,
        _count_antigravity,
    )


def _protected_count_mapping_antigravity(value: object) -> MappingProxyType:
    return _mapping_antigravity(
        value,
        "protected_broken",
        FROZEN_CONFIG_ANTIGRAVITY.change_types[:3],
        _count_antigravity,
    )


@dataclass(frozen=True, slots=True)
class Exp471FoldResultAntigravity:
    seed: int
    fold_index: int
    c0_per_type_auc: Mapping[str, float]
    c12_per_type_auc: Mapping[str, float]
    pair_repairs: Mapping[str, PairRepairCountsAntigravity]
    normal_confirmations: int
    fallback_exact: bool
    finite: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed must be one of the two frozen Exp471 seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_index must be one of the five frozen folds")
        object.__setattr__(
            self,
            "c0_per_type_auc",
            _auc_mapping_antigravity(self.c0_per_type_auc, "c0_per_type_auc"),
        )
        object.__setattr__(
            self,
            "c12_per_type_auc",
            _auc_mapping_antigravity(self.c12_per_type_auc, "c12_per_type_auc"),
        )
        object.__setattr__(self, "pair_repairs", _pair_repairs_mapping_antigravity(self.pair_repairs))
        _count_antigravity(self.normal_confirmations, "normal_confirmations")
        _bool_antigravity(self.fallback_exact, "fallback_exact")
        _bool_antigravity(self.finite, "finite")
        _bool_antigravity(self.passed, "passed")
        if type(self.failure_reasons) is not tuple or any(
            type(reason) is not str for reason in self.failure_reasons
        ):
            raise ValueError("failure_reasons must be a tuple of strings")


@dataclass(frozen=True, slots=True)
class Exp471QualificationAntigravity:
    fold_results: tuple[Exp471FoldResultAntigravity, ...]
    period_fixed: Mapping[int, int]
    period_broken: Mapping[int, int]
    protected_broken: Mapping[str, int]
    normal_confirmations: Mapping[int, int]
    fallback_exact: bool
    finite: bool
    label_query_count: int
    artifact_map_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.fold_results) is not tuple:
            raise ValueError("fold_results must be an immutable tuple")
        expected_identities = tuple(
            (seed, fold_index)
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if len(self.fold_results) != len(expected_identities):
            raise ValueError("fold_results must contain exactly ten seed-fold rows")
        if any(type(row) is not Exp471FoldResultAntigravity for row in self.fold_results):
            raise ValueError("fold_results must contain exact Exp471FoldResultAntigravity values")
        if tuple((row.seed, row.fold_index) for row in self.fold_results) != expected_identities:
            raise ValueError("fold_results must preserve frozen seed/fold order")
        object.__setattr__(self, "period_fixed", _seed_count_mapping_antigravity(self.period_fixed, "period_fixed"))
        object.__setattr__(self, "period_broken", _seed_count_mapping_antigravity(self.period_broken, "period_broken"))
        object.__setattr__(self, "protected_broken", _protected_count_mapping_antigravity(self.protected_broken))
        object.__setattr__(
            self,
            "normal_confirmations",
            _seed_count_mapping_antigravity(self.normal_confirmations, "normal_confirmations"),
        )
        _bool_antigravity(self.fallback_exact, "fallback_exact")
        _bool_antigravity(self.finite, "finite")
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("label_query_count must remain the exact pre-run value 0")
        if self.artifact_map_sha256 is not None and (
            type(self.artifact_map_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.artifact_map_sha256) is None
        ):
            raise ValueError("artifact_map_sha256 must be None or a lowercase 64-hex SHA-256")


__all__ = (
    "PairRepairCountsAntigravity",
    "Exp471FoldResultAntigravity",
    "Exp471QualificationAntigravity",
)
