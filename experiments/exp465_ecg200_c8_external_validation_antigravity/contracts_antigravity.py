"""Immutable score/result and initial-governance contracts for Exp465."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
import re
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_SYNTHETIC_TYPE_COUNT_ANTIGRAVITY = 4
_MEAN_TOLERANCE_ANTIGRAVITY = 1e-12


def _int_antigravity(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex digest")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _exact_tuple_antigravity(
    value: object,
    name: str,
    item_validator,
) -> tuple[object, ...]:
    if type(value) is not tuple or not value:
        raise ValueError(f"{name} must be a non-empty tuple")
    return tuple(
        item_validator(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class Exp465MethodScoreContractAntigravity:
    method: str
    scores: tuple[float, ...]
    confirmations: tuple[int, ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.method) is not str
            or self.method not in FROZEN_CONFIG_ANTIGRAVITY.method_order
        ):
            raise ValueError("method must be one of the frozen C0/C6/C8 IDs")
        _exact_tuple_antigravity(
            self.scores,
            "scores",
            lambda value, name: _unit_float_antigravity(value, name),
        )
        _exact_tuple_antigravity(
            self.confirmations,
            "confirmations",
            lambda value, name: _int_antigravity(value, name),
        )
        if len(self.scores) != len(self.confirmations):
            raise ValueError("scores and confirmations must have equal length")
        _sha64_antigravity(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True, slots=True)
class Exp465FoldResultContractAntigravity:
    seed: int
    fold_index: int
    per_type_auc: tuple[float, float, float, float]
    mean_auc: float
    c0_gradual_auc: float
    c0_period_auc: float
    c8_gradual_auc: float
    c8_period_auc: float
    all_scores_finite: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds
        ):
            raise ValueError("seed must be one of the two frozen Exp465 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index must be one of the five frozen folds")
        auc_values = _exact_tuple_antigravity(
            self.per_type_auc,
            "per_type_auc",
            lambda value, name: _unit_float_antigravity(value, name),
        )
        if len(auc_values) != _SYNTHETIC_TYPE_COUNT_ANTIGRAVITY:
            raise ValueError("per_type_auc must contain exactly four synthetic-type AUCs")
        _unit_float_antigravity(self.mean_auc, "mean_auc")
        expected_mean = fsum(auc_values) / len(auc_values)
        if abs(self.mean_auc - expected_mean) > _MEAN_TOLERANCE_ANTIGRAVITY:
            raise ValueError("mean_auc differs from the four per-type AUCs")
        for name in (
            "c0_gradual_auc",
            "c0_period_auc",
            "c8_gradual_auc",
            "c8_period_auc",
        ):
            _unit_float_antigravity(getattr(self, name), name)
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        _bool_antigravity(self.passed, "passed")
        reasons = self.failure_reasons
        if (
            type(reasons) is not tuple
            or any(type(item) is not str or not item for item in reasons)
        ):
            raise ValueError("failure_reasons must be a tuple of non-empty strings")
        if self.passed is (len(reasons) != 0):
            raise ValueError("passed and failure_reasons disagree")


def _fold_result_antigravity(value: object) -> Exp465FoldResultContractAntigravity:
    if type(value) is not Exp465FoldResultContractAntigravity:
        raise ValueError("rows must contain exact Exp465FoldResultContractAntigravity values")
    return value


def _seed_confirmations_antigravity(
    value: Mapping[int, int] | object,
) -> MappingProxyType[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError("seed_normal_confirmations must be a mapping")
    expected_seeds = FROZEN_CONFIG_ANTIGRAVITY.seeds
    if tuple(value) != expected_seeds:
        raise ValueError("seed_normal_confirmations must preserve frozen seed order")
    frozen = {
        seed: _int_antigravity(value[seed], f"seed_normal_confirmations[{seed}]")
        for seed in expected_seeds
    }
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class Exp465QualificationContractAntigravity:
    rows: tuple[Exp465FoldResultContractAntigravity, ...]
    seed_normal_confirmations: Mapping[int, int]
    label_query_count: int
    official_evaluate_count: int
    artifact_map_sha256: None

    def __post_init__(self) -> None:
        rows = self.rows
        if type(rows) is not tuple:
            raise ValueError("rows must be an immutable tuple")
        rows = tuple(_fold_result_antigravity(row) for row in rows)
        object.__setattr__(self, "rows", rows)
        expected_rows = FROZEN_CONFIG_ANTIGRAVITY.required_stability_records
        if len(rows) != expected_rows:
            raise ValueError(f"rows must contain exactly {expected_rows} stability records")
        object.__setattr__(
            self,
            "seed_normal_confirmations",
            _seed_confirmations_antigravity(self.seed_normal_confirmations),
        )
        _int_antigravity(self.label_query_count, "label_query_count")
        _int_antigravity(self.official_evaluate_count, "official_evaluate_count")
        if self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("label_query_count differs from the frozen governance truth")
        if (
            self.official_evaluate_count
            != FROZEN_CONFIG_ANTIGRAVITY.official_evaluate_count_must_remain
        ):
            raise ValueError("official_evaluate_count differs from the frozen governance truth")
        if self.artifact_map_sha256 is not None:
            raise ValueError("artifact_map_sha256 must initially be null")


def initial_governance_state_antigravity() -> MappingProxyType[str, object]:
    """Return a fresh immutable pre-run governance snapshot."""

    return MappingProxyType(
        {
            "label_query_count": FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
            "official_evaluate_count": (
                FROZEN_CONFIG_ANTIGRAVITY.official_evaluate_count_must_remain
            ),
            "artifact_map_sha256": None,
        }
    )


__all__ = (
    "Exp465FoldResultContractAntigravity",
    "Exp465MethodScoreContractAntigravity",
    "Exp465QualificationContractAntigravity",
    "initial_governance_state_antigravity",
)
