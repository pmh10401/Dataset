"""Immutable fold and qualification containers for Exp468 Task 1.

Both AUC tuples are positionally bound to ``FROZEN_CONFIG_ANTIGRAVITY.change_types``.
Its fourth value, ``period_change``, is the exact Exp466 synthetic scope that
Task 2 will bind. AUC computation and the final scientific gate belong to
Task 4; this module enforces only types, ranges, order, and immutability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_AUC_COUNT_ANTIGRAVITY = len(FROZEN_CONFIG_ANTIGRAVITY.change_types)


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _count_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _auc_tuple_antigravity(value: object, name: str) -> tuple[float, float, float, float]:
    if type(value) is not tuple or len(value) != _AUC_COUNT_ANTIGRAVITY:
        raise ValueError(f"{name} must be an immutable four-value tuple")
    return tuple(  # type: ignore[return-value]
        _unit_float_antigravity(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True, slots=True)
class Exp468FoldResultAntigravity:
    """One typed fold result; scientific pass/fail calculation remains Task 4."""

    seed: int
    fold_index: int
    c0_per_type_auc: tuple[float, float, float, float]
    c10_per_type_auc: tuple[float, float, float, float]
    period_confirmations: int
    normal_confirmations: int
    all_scores_finite: bool
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed must be one of the two frozen Exp468 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index must be one of the five frozen folds")
        _auc_tuple_antigravity(self.c0_per_type_auc, "c0_per_type_auc")
        _auc_tuple_antigravity(self.c10_per_type_auc, "c10_per_type_auc")
        _count_antigravity(self.period_confirmations, "period_confirmations")
        _count_antigravity(self.normal_confirmations, "normal_confirmations")
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        passed = _bool_antigravity(self.passed, "passed")
        reasons = self.failure_reasons
        if (
            type(reasons) is not tuple
            or any(type(item) is not str or not item for item in reasons)
        ):
            raise ValueError("failure_reasons must be an immutable tuple of non-empty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("failure_reasons must contain unique strings")
        if passed is not (len(reasons) == 0):
            raise ValueError("passed and failure_reasons disagree")


def _frozen_seed_counts_antigravity(
    value: Mapping[int, int] | object,
    name: str,
    *,
    require_all_trials: bool,
) -> MappingProxyType[int, int]:
    if type(value) is not MappingProxyType:
        raise ValueError(f"{name} must be an immutable mapping")
    if tuple(value) != FROZEN_CONFIG_ANTIGRAVITY.seeds:
        raise ValueError(f"{name} must preserve frozen seed order")
    frozen: dict[int, int] = {}
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        count = _count_antigravity(value[seed], f"{name}[{seed}]")
        if require_all_trials:
            if count != FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed:
                raise ValueError("normal_trial_counts must contain exactly 69 trials per seed")
        elif count > FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed:
            raise ValueError("normal confirmation counts must be between 0 and 69")
        frozen[seed] = count
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class Exp468QualificationAntigravity:
    fold_results: tuple[Exp468FoldResultAntigravity, ...]
    normal_trial_counts: Mapping[int, int]
    normal_confirmations: Mapping[int, int]
    label_query_count: int
    artifact_map_sha256: None

    def __post_init__(self) -> None:
        if type(self.fold_results) is not tuple:
            raise ValueError("fold_results must be an immutable tuple")
        if len(self.fold_results) != FROZEN_CONFIG_ANTIGRAVITY.required_fold_results:
            raise ValueError("fold_results must contain exactly ten seed-fold rows")
        if any(type(row) is not Exp468FoldResultAntigravity for row in self.fold_results):
            raise ValueError("fold_results must contain exact Exp468FoldResultAntigravity values")
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
                require_all_trials=True,
            ),
        )
        object.__setattr__(
            self,
            "normal_confirmations",
            _frozen_seed_counts_antigravity(
                self.normal_confirmations,
                "normal_confirmations",
                require_all_trials=False,
            ),
        )
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("label_query_count must remain the exact pre-run value 0")
        if self.artifact_map_sha256 is not None:
            raise ValueError("artifact_map_sha256 must remain null before a private run")


__all__ = (
    "Exp468FoldResultAntigravity",
    "Exp468QualificationAntigravity",
)
