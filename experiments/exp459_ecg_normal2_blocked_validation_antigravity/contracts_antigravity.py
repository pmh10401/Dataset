"""Immutable validated blocked-validation contracts for Exp459 Task 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from math import sqrt
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_WILSON_95_Z_ANTIGRAVITY = 1.959963984540054


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


def _seed_antigravity(value: object) -> int:
    if type(value) is not int or value not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the two exact Hydra seeds")
    return value


def _fold_index_antigravity(value: object) -> int:
    if type(value) is not int or not 0 <= value < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be an exact Task2 fold index")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _readonly_owned_indices_antigravity(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.ndim != 1 or value.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if value.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must contain exact integer indices")
    indices = np.array(value, dtype=np.int64, copy=True)
    if np.any(indices < 0):
        raise ValueError(f"{name} must be non-negative")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} must be unique")
    if not np.array_equal(indices, np.sort(indices)):
        raise ValueError(f"{name} must be sorted")
    indices.flags.writeable = False
    return indices


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


def wilson_upper_antigravity(successes: int, trials: int) -> float:
    _positive_int_antigravity(trials, "normal_trials")
    _bounded_int_antigravity(successes, "normal_alerts", upper=trials)
    proportion = successes / trials
    z_squared = _WILSON_95_Z_ANTIGRAVITY**2
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    adjustment = _WILSON_95_Z_ANTIGRAVITY * sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * trials)) / trials
    )
    return min(1.0, (center + adjustment) / denominator)


def _expected_query_indices_antigravity(fold_index: int) -> np.ndarray:
    start, stop = FROZEN_CONFIG_ANTIGRAVITY.query_ranges[fold_index]
    indices = np.arange(start, stop + 1, dtype=np.int64)
    indices.flags.writeable = False
    return indices


def _expected_fit_indices_antigravity(fold_index: int) -> np.ndarray:
    query = _expected_query_indices_antigravity(fold_index)
    fit = np.concatenate(
        (
            np.arange(0, int(query[0]), dtype=np.int64),
            np.arange(int(query[-1]) + 1, FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows, dtype=np.int64),
        )
    )
    fit.flags.writeable = False
    return fit


def _fold_signature_antigravity(value: object) -> tuple[object, ...]:
    if type(value) is not FoldEvidenceAntigravity:
        raise ValueError("fold record type is invalid")
    return (
        value.dataset,
        value.seed,
        value.fold_index,
        value.fit_indices.tobytes(),
        value.query_indices.tobytes(),
        value.normal_trials,
        value.normal_alerts,
        value.normal_wilson_upper,
        tuple(value.per_type_auc[name] for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types),
        value.pooled_auc,
        value.passed,
    )


@dataclass(frozen=True, slots=True)
class BlockedFoldAntigravity:
    fold_index: int
    fit_indices: np.ndarray
    query_indices: np.ndarray

    def __post_init__(self) -> None:
        _fold_index_antigravity(self.fold_index)
        fit_indices = _readonly_owned_indices_antigravity(self.fit_indices, "fit_indices")
        query_indices = _readonly_owned_indices_antigravity(self.query_indices, "query_indices")
        if np.intersect1d(fit_indices, query_indices).size:
            raise ValueError("fit_indices and query_indices must remain disjoint")
        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "query_indices", query_indices)


@dataclass(frozen=True, slots=True)
class FoldEvidenceAntigravity:
    dataset: str
    seed: int
    fold_index: int
    fit_indices: np.ndarray
    query_indices: np.ndarray
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    per_type_auc: Mapping[str, float]
    pooled_auc: float
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        _seed_antigravity(self.seed)
        fold_index = _fold_index_antigravity(self.fold_index)
        fit_indices = _readonly_owned_indices_antigravity(self.fit_indices, "fit_indices")
        query_indices = _readonly_owned_indices_antigravity(self.query_indices, "query_indices")
        if np.intersect1d(fit_indices, query_indices).size:
            raise ValueError("fit_indices and query_indices must remain disjoint")
        if not np.array_equal(query_indices, _expected_query_indices_antigravity(fold_index)):
            raise ValueError("query_indices must match the exact blocked fold range")
        if not np.array_equal(fit_indices, _expected_fit_indices_antigravity(fold_index)):
            raise ValueError("fit_indices must match the exact blocked fold complement")
        _positive_int_antigravity(self.normal_trials, "normal_trials")
        if self.normal_trials != len(query_indices):
            raise ValueError("normal_trials must match the blocked query count")
        _bounded_int_antigravity(self.normal_alerts, "normal_alerts", upper=self.normal_trials)
        expected_wilson = wilson_upper_antigravity(self.normal_alerts, self.normal_trials)
        if _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper") != expected_wilson:
            raise ValueError("normal_wilson_upper differs from semantic recomputation")
        per_type_auc = _validate_per_type_auc_antigravity(self.per_type_auc)
        expected_pooled = float(
            sum(per_type_auc[name] for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types)
            / len(FROZEN_CONFIG_ANTIGRAVITY.primary_types)
        )
        if _unit_float_antigravity(self.pooled_auc, "pooled_auc") != expected_pooled:
            raise ValueError("pooled_auc differs from semantic recomputation")
        expected_passed = bool(
            expected_wilson <= FROZEN_CONFIG_ANTIGRAVITY.normal_wilson_upper_max
            and expected_pooled >= FROZEN_CONFIG_ANTIGRAVITY.pooled_auc_min
            and all(
                per_type_auc[name] >= FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min
                for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
            )
        )
        if _bool_antigravity(self.passed, "passed") is not expected_passed:
            raise ValueError("passed differs from semantic recomputation")
        object.__setattr__(self, "fit_indices", fit_indices)
        object.__setattr__(self, "query_indices", query_indices)
        object.__setattr__(self, "per_type_auc", per_type_auc)


@dataclass(frozen=True, slots=True)
class SeedSummaryAntigravity:
    dataset: str
    seed: int
    fold_records: tuple[FoldEvidenceAntigravity, ...]
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        seed = _seed_antigravity(self.seed)
        if type(self.fold_records) is not tuple or len(self.fold_records) != FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_records must be the exact five-record tuple")
        if any(type(record) is not FoldEvidenceAntigravity for record in self.fold_records):
            raise ValueError("fold_records must contain FoldEvidenceAntigravity records")
        if any(record.dataset != self.dataset or record.seed != seed for record in self.fold_records):
            raise ValueError("fold_records must match the summary dataset and seed")
        if tuple(record.fold_index for record in self.fold_records) != tuple(
            range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        ):
            raise ValueError("fold_records must remain in exact blocked-fold order")
        expected_passed = all(record.passed for record in self.fold_records)
        if _bool_antigravity(self.passed, "passed") is not expected_passed:
            raise ValueError("seed summary differs from fold pass recomputation")


@dataclass(frozen=True, slots=True)
class BlockedQualificationAntigravity:
    dataset: str
    fold_evidence: tuple[FoldEvidenceAntigravity, ...]
    seed_summaries: tuple[SeedSummaryAntigravity, SeedSummaryAntigravity]
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.fold_evidence) is not tuple or len(self.fold_evidence) != (
            len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_evidence must be the exact ten-record tuple")
        if any(type(record) is not FoldEvidenceAntigravity for record in self.fold_evidence):
            raise ValueError("fold_evidence must contain FoldEvidenceAntigravity records")
        expected_order = tuple(
            (seed, fold_index)
            for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if tuple((record.seed, record.fold_index) for record in self.fold_evidence) != expected_order:
            raise ValueError("fold_evidence must stay in exact seed-then-fold order")
        if any(record.dataset != self.dataset for record in self.fold_evidence):
            raise ValueError("fold_evidence datasets must match qualification dataset")
        if type(self.seed_summaries) is not tuple or len(self.seed_summaries) != len(
            FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        ):
            raise ValueError("seed_summaries must be the exact two-record tuple")
        if any(type(summary) is not SeedSummaryAntigravity for summary in self.seed_summaries):
            raise ValueError("seed_summaries must contain SeedSummaryAntigravity records")
        if tuple(summary.seed for summary in self.seed_summaries) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed_summaries must stay in exact Hydra seed order")
        if any(summary.dataset != self.dataset for summary in self.seed_summaries):
            raise ValueError("seed_summaries datasets must match qualification dataset")
        for summary_index, summary in enumerate(self.seed_summaries):
            start = summary_index * FROZEN_CONFIG_ANTIGRAVITY.folds
            stop = start + FROZEN_CONFIG_ANTIGRAVITY.folds
            if tuple(_fold_signature_antigravity(record) for record in summary.fold_records) != tuple(
                _fold_signature_antigravity(record) for record in self.fold_evidence[start:stop]
            ):
                raise ValueError("seed_summaries must mirror the ordered fold_evidence slices")
        expected_passed = all(summary.passed for summary in self.seed_summaries)
        if _bool_antigravity(self.passed, "passed") is not expected_passed:
            raise ValueError("qualification differs from both-seed recomputation")


@dataclass(frozen=True, slots=True)
class ValidationDecisionAntigravity:
    dataset: str
    qualification: BlockedQualificationAntigravity
    status: str
    passed: bool

    def __post_init__(self) -> None:
        _dataset_antigravity(self.dataset)
        if type(self.qualification) is not BlockedQualificationAntigravity:
            raise ValueError("qualification type is invalid")
        if self.qualification.dataset != self.dataset:
            raise ValueError("qualification dataset must match the decision dataset")
        if type(self.status) is not str or self.status not in {"stable", "unstable"}:
            raise ValueError("status must be stable or unstable")
        expected_passed = self.qualification.passed
        if _bool_antigravity(self.passed, "passed") is not expected_passed:
            raise ValueError("decision passed differs from qualification recomputation")
        expected_status = "stable" if expected_passed else "unstable"
        if self.status != expected_status:
            raise ValueError("status differs from qualification recomputation")


__all__ = (
    "BlockedFoldAntigravity",
    "BlockedQualificationAntigravity",
    "FoldEvidenceAntigravity",
    "SeedSummaryAntigravity",
    "ValidationDecisionAntigravity",
    "wilson_upper_antigravity",
)
