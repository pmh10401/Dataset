"""Immutable phase evidence, fold result, and qualification contracts for Exp466."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum, isfinite
import re
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_AUC_COUNT_ANTIGRAVITY = 4
_FLOAT_TOLERANCE_ANTIGRAVITY = 1e-12


def _finite_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    result = _finite_float_antigravity(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _int_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class Exp466PhaseEvidenceAntigravity:
    identity_error: float
    undo_error: float
    margin: float
    fit_normal_max_margin: float
    confirmed: bool
    evidence_sha256: str

    def __post_init__(self) -> None:
        identity_error = _finite_float_antigravity(self.identity_error, "identity_error")
        undo_error = _finite_float_antigravity(self.undo_error, "undo_error")
        margin = _finite_float_antigravity(self.margin, "margin")
        threshold = _finite_float_antigravity(
            self.fit_normal_max_margin,
            "fit_normal_max_margin",
        )
        if identity_error < 0.0 or undo_error < 0.0:
            raise ValueError("phase errors must be non-negative")
        if abs(margin - (identity_error - undo_error)) > _FLOAT_TOLERANCE_ANTIGRAVITY:
            raise ValueError("margin must equal identity_error - undo_error")
        confirmed = _bool_antigravity(self.confirmed, "confirmed")
        if confirmed is not (margin > 0.0 and margin > threshold):
            raise ValueError("confirmed must match the frozen positive strict-margin rule")
        _sha64_antigravity(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True, slots=True)
class Exp466FoldResultAntigravity:
    seed: int
    fold_index: int
    per_type_auc: tuple[float, float, float, float]
    mean_auc: float
    c0_gradual_auc: float
    c0_period_auc: float
    c9_gradual_auc: float
    c9_period_auc: float
    period_confirmations: int
    all_scores_finite: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed must be one of the two frozen Exp466 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index must be one of the five frozen folds")
        if type(self.per_type_auc) is not tuple or len(self.per_type_auc) != _AUC_COUNT_ANTIGRAVITY:
            raise ValueError("per_type_auc must be an immutable four-value tuple")
        auc_values = tuple(
            _unit_float_antigravity(value, f"per_type_auc[{index}]")
            for index, value in enumerate(self.per_type_auc)
        )
        mean_auc = _unit_float_antigravity(self.mean_auc, "mean_auc")
        if abs(mean_auc - fsum(auc_values) / _AUC_COUNT_ANTIGRAVITY) > _FLOAT_TOLERANCE_ANTIGRAVITY:
            raise ValueError("mean_auc differs from the four per-type AUCs")
        for name in (
            "c0_gradual_auc",
            "c0_period_auc",
            "c9_gradual_auc",
            "c9_period_auc",
        ):
            _unit_float_antigravity(getattr(self, name), name)
        period_confirmations = _int_antigravity(
            self.period_confirmations,
            "period_confirmations",
        )
        finite = _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        passed = _bool_antigravity(self.passed, "passed")
        reasons = self.failure_reasons
        if (
            type(reasons) is not tuple
            or any(type(item) is not str or not item for item in reasons)
        ):
            raise ValueError("failure_reasons must be a tuple of non-empty strings")
        if passed is (len(reasons) != 0):
            raise ValueError("passed and failure_reasons disagree")
        gate_passed = (
            finite
            and all(value >= FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min for value in auc_values)
            and mean_auc >= FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min
            and self.c9_period_auc > self.c0_period_auc
            and self.c9_gradual_auc >= self.c0_gradual_auc
            and period_confirmations >= 1
        )
        if passed and not gate_passed:
            raise ValueError("passed fold result violates a frozen C9 gate")


def _frozen_seed_counts_antigravity(
    value: Mapping[int, int] | object,
    name: str,
    expected_count: int,
) -> MappingProxyType[int, int]:
    if type(value) is not MappingProxyType:
        raise ValueError(f"{name} must be an immutable mapping")
    if tuple(value) != FROZEN_CONFIG_ANTIGRAVITY.seeds:
        raise ValueError(f"{name} must preserve frozen seed order")
    frozen: dict[int, int] = {}
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        count = _int_antigravity(value[seed], f"{name}[{seed}]")
        if count != expected_count:
            if name == "normal_trial_counts":
                raise ValueError("normal_trial_counts must contain exactly 69 trials per seed")
            raise ValueError("normal confirmation counts must remain exactly zero")
        frozen[seed] = count
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class Exp466QualificationAntigravity:
    fold_results: tuple[Exp466FoldResultAntigravity, ...]
    normal_trial_counts: Mapping[int, int]
    normal_confirmations: Mapping[int, int]
    label_query_count: int
    official_evaluate_count: int
    official_inspect_count: int
    artifact_map_sha256: None

    def __post_init__(self) -> None:
        if type(self.fold_results) is not tuple:
            raise ValueError("fold_results must be an immutable tuple")
        if len(self.fold_results) != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
            raise ValueError("fold_results must contain exactly ten seed-fold rows")
        if any(type(row) is not Exp466FoldResultAntigravity for row in self.fold_results):
            raise ValueError("fold_results must contain exact Exp466FoldResultAntigravity values")
        expected_keys = tuple(
            (seed, fold_index)
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if tuple((row.seed, row.fold_index) for row in self.fold_results) != expected_keys:
            raise ValueError("fold_results must preserve frozen seed/fold order")
        object.__setattr__(
            self,
            "normal_trial_counts",
            _frozen_seed_counts_antigravity(
                self.normal_trial_counts,
                "normal_trial_counts",
                FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed,
            ),
        )
        object.__setattr__(
            self,
            "normal_confirmations",
            _frozen_seed_counts_antigravity(
                self.normal_confirmations,
                "normal confirmations",
                0,
            ),
        )
        for name, expected in (
            ("label_query_count", FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain),
            (
                "official_evaluate_count",
                FROZEN_CONFIG_ANTIGRAVITY.official_evaluate_count_must_remain,
            ),
            (
                "official_inspect_count",
                FROZEN_CONFIG_ANTIGRAVITY.official_inspect_count_must_remain,
            ),
        ):
            actual = _int_antigravity(getattr(self, name), name)
            if actual != expected:
                raise ValueError(f"{name} differs from frozen zero-use governance truth")
        if self.artifact_map_sha256 is not None:
            raise ValueError("artifact_map_sha256 must remain null before a private run")


__all__ = (
    "Exp466FoldResultAntigravity",
    "Exp466PhaseEvidenceAntigravity",
    "Exp466QualificationAntigravity",
)
