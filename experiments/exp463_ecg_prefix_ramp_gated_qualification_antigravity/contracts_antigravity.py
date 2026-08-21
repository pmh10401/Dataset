"""Immutable Task 2 contracts for Exp463 and the frozen Exp462 snapshot.

The future metric containers describe shape and ordering for later tasks only;
no score calculation, evaluation, or publication is performed here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
import re
from types import MappingProxyType

from .config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
    HistoricalPeriodThresholdAntigravity,
)


_METHOD_IDS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_order
_EXP462_METHODS_ANTIGRAVITY = (
    "hydra_period_rank_max",
    "gradual_signature_local_gap",
    "hydra_period_gradual_rank_max",
)
_LEAF_ORDER_ANTIGRAVITY = tuple(FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256)


def _int_antigravity(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _seed_antigravity(value: object) -> int:
    if type(value) is not int or value not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the frozen seeds")
    return value


def _fold_antigravity(value: object) -> int:
    if type(value) is not int or not 0 <= value < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be one of the frozen blocked folds")
    return value


def _unit_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex digest")
    return value


def _score_tuple_antigravity(value: object, name: str) -> tuple[float, ...]:
    if type(value) not in (tuple, list):
        raise ValueError(f"{name} must be a tuple/list score vector")
    try:
        result = tuple(float(item) for item in value)  # type: ignore[union-attr]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _per_type_auc_antigravity(value: Mapping[str, object], name: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or tuple(value) != _CHANGE_TYPES_ANTIGRAVITY:
        raise ValueError(f"{name} must preserve the exact ordered change types")
    frozen = {
        change_type: _unit_float_antigravity(value[change_type], f"{name}[{change_type}]")
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    return MappingProxyType(frozen)


def _method_id_antigravity(value: object) -> str:
    if type(value) is not str or value not in _METHOD_IDS_ANTIGRAVITY:
        raise ValueError("method must be one of frozen M0-M6 IDs")
    return value


def _reason_tuple_antigravity(value: object) -> tuple[str, ...]:
    if type(value) is not tuple or any(type(item) is not str or not item for item in value):
        raise ValueError("failure_reasons must be a tuple of non-empty strings")
    return value


@dataclass(frozen=True, slots=True)
class Exp463HistoricalPeriodFoldAntigravity:
    seed: int
    fold_index: int
    period_change_auc: float

    def __post_init__(self) -> None:
        _seed_antigravity(self.seed)
        _fold_antigravity(self.fold_index)
        _unit_float_antigravity(self.period_change_auc, "period_change_auc")


@dataclass(frozen=True, slots=True)
class Exp462ResultRowIdentityAntigravity:
    method: str
    seed: int
    fold_index: int

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _EXP462_METHODS_ANTIGRAVITY:
            raise ValueError("Exp462 row method is not frozen")
        _seed_antigravity(self.seed)
        _fold_antigravity(self.fold_index)


@dataclass(frozen=True, slots=True)
class Exp462MethodSummaryAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_gradual_drift_auc: float
    min_period_change_auc: float
    min_pooled_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool
    passed: bool

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in _EXP462_METHODS_ANTIGRAVITY:
            raise ValueError("Exp462 summary method is not frozen")
        fold_count = _int_antigravity(self.fold_count, "fold_count", positive=True)
        passed_count = _int_antigravity(self.passed_count, "passed_count")
        failed_count = _int_antigravity(self.failed_count, "failed_count")
        if fold_count != passed_count + failed_count:
            raise ValueError("summary counts must cover all folds")
        _unit_float_antigravity(self.min_gradual_drift_auc, "min_gradual_drift_auc")
        _unit_float_antigravity(self.min_period_change_auc, "min_period_change_auc")
        _unit_float_antigravity(self.min_pooled_auc, "min_pooled_auc")
        _unit_float_antigravity(self.max_normal_wilson_upper, "max_normal_wilson_upper")
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class Exp462ResultSnapshotAntigravity:
    row_count: int
    failed_row_count: int
    summary_count: int
    leaf_count: int
    rows: tuple[Exp462ResultRowIdentityAntigravity, ...]
    summaries: tuple[Exp462MethodSummaryAntigravity, ...]
    primary_candidate: str
    primary_passed_count: int
    primary_required_count: int
    primary_min_per_type_auc: Mapping[str, float]
    primary_min_pooled_auc: float
    primary_max_normal_wilson_upper: float
    leaf_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.row_count != 30 or self.failed_row_count != 18:
            raise ValueError("Exp462 result row inventory differs from exact 30/18 snapshot")
        if self.summary_count != 3 or self.leaf_count != 7:
            raise ValueError("Exp462 result inventory differs from exact 3/7 snapshot")
        if type(self.rows) is not tuple or len(self.rows) != self.row_count:
            raise ValueError("rows must contain exact 30 identities")
        expected_rows = tuple(
            Exp462ResultRowIdentityAntigravity(method, seed, fold_index)
            for method in _EXP462_METHODS_ANTIGRAVITY
            for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if self.rows != expected_rows:
            raise ValueError("Exp462 row order differs from method/seed/fold order")
        if type(self.summaries) is not tuple or tuple(row.method for row in self.summaries) != _EXP462_METHODS_ANTIGRAVITY:
            raise ValueError("summaries must preserve Exp462 method order")
        if len(self.summaries) != self.summary_count:
            raise ValueError("summary count mismatch")
        if self.primary_candidate != "hydra_period_gradual_rank_max":
            raise ValueError("primary candidate differs from Exp462 result")
        if self.primary_passed_count != 3 or self.primary_required_count != 10:
            raise ValueError("primary count differs from Exp462 result")
        per_type = _per_type_auc_antigravity(self.primary_min_per_type_auc, "primary_min_per_type_auc")
        if per_type != MappingProxyType(dict(FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_min_per_type_auc)):
            raise ValueError("primary per-type minima differ from Exp462 result")
        _unit_float_antigravity(self.primary_min_pooled_auc, "primary_min_pooled_auc")
        _unit_float_antigravity(self.primary_max_normal_wilson_upper, "primary_max_normal_wilson_upper")
        if not isinstance(self.leaf_sha256, Mapping) or tuple(self.leaf_sha256) != _LEAF_ORDER_ANTIGRAVITY:
            raise ValueError("leaf_sha256 must preserve exact seven-leaf order")
        frozen = {
            name: _sha64_antigravity(self.leaf_sha256[name], f"leaf_sha256[{name}]")
            for name in _LEAF_ORDER_ANTIGRAVITY
        }
        if frozen != dict(FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256):
            raise ValueError("Exp462 leaf hashes differ from frozen snapshot")
        object.__setattr__(self, "primary_min_per_type_auc", per_type)
        object.__setattr__(self, "leaf_sha256", MappingProxyType(frozen))

    @property
    def leaf_inventory(self) -> tuple[tuple[str, str], ...]:
        return tuple(self.leaf_sha256.items())

    @property
    def method_order(self) -> tuple[str, ...]:
        return _EXP462_METHODS_ANTIGRAVITY

    @property
    def primary_summary(self) -> Exp462MethodSummaryAntigravity:
        return self.summaries[-1]


@dataclass(frozen=True, slots=True)
class Exp462AuthorityAntigravity:
    predecessor_experiment: str
    reviewed_code_commit: str
    terminal_closeout_commit: str
    status: str
    primary_candidate: str
    passed_record_count: int
    required_record_count: int
    failed_gradual_drift_auc: float
    period_min_auc: float
    historical_period_folds: tuple[Exp463HistoricalPeriodFoldAntigravity, ...]
    result_snapshot: Exp462ResultSnapshotAntigravity
    label_query_count: int
    private_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.predecessor_experiment != FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment:
            raise ValueError("predecessor experiment differs")
        if type(self.reviewed_code_commit) is not str or re.fullmatch(r"[0-9a-f]{40}", self.reviewed_code_commit) is None:
            raise ValueError("reviewed_code_commit must be the exact 40-hex commit")
        if self.reviewed_code_commit != FROZEN_CONFIG_ANTIGRAVITY.predecessor_reviewed_code_commit:
            raise ValueError("reviewed_code_commit differs from frozen Exp462 authority")
        if self.terminal_closeout_commit != FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_closeout_commit:
            raise ValueError("terminal closeout commit differs")
        if self.status != FROZEN_CONFIG_ANTIGRAVITY.predecessor_status:
            raise ValueError("status differs")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate:
            raise ValueError("primary candidate differs")
        if self.passed_record_count != FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count:
            raise ValueError("passed record count differs")
        if self.required_record_count != FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count:
            raise ValueError("required record count differs")
        if self.failed_gradual_drift_auc != FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc:
            raise ValueError("failed gradual AUC differs")
        if self.period_min_auc != FROZEN_CONFIG_ANTIGRAVITY.predecessor_period_min_auc:
            raise ValueError("period minimum differs")
        expected_folds = tuple(
            Exp463HistoricalPeriodFoldAntigravity(row.seed, row.fold_index, row.period_change_auc_min)
            for row in FROZEN_CONFIG_ANTIGRAVITY.predecessor_historical_period_folds
        )
        if self.historical_period_folds != expected_folds:
            raise ValueError("historical period folds differ")
        if self.result_snapshot != Exp462ResultSnapshotAntigravity(
            row_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_row_count,
            failed_row_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_failed_row_count,
            summary_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_summary_count,
            leaf_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_leaf_count,
            rows=self.result_snapshot.rows,
            summaries=self.result_snapshot.summaries,
            primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate,
            primary_passed_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count,
            primary_required_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count,
            primary_min_per_type_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_min_per_type_auc,
            primary_min_pooled_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_min_pooled_auc,
            primary_max_normal_wilson_upper=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_max_normal_wilson_upper,
            leaf_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256,
        ):
            raise ValueError("result snapshot differs from frozen Exp462 result")
        if self.label_query_count != FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain:
            raise ValueError("label query count must remain zero")
        if not isinstance(self.private_sha256, Mapping) or tuple(self.private_sha256) != _LEAF_ORDER_ANTIGRAVITY:
            raise ValueError("private_sha256 must preserve exact leaf order")
        private = {
            name: _sha64_antigravity(self.private_sha256[name], f"private_sha256[{name}]")
            for name in _LEAF_ORDER_ANTIGRAVITY
        }
        if private != dict(FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256):
            raise ValueError("private hashes differ from frozen Exp462 result")
        object.__setattr__(self, "private_sha256", MappingProxyType(private))

    @property
    def row_count(self) -> int:
        return self.result_snapshot.row_count

    @property
    def summary_count(self) -> int:
        return self.result_snapshot.summary_count

    @property
    def leaf_count(self) -> int:
        return self.result_snapshot.leaf_count

    @property
    def artifact_map_sha256(self) -> str:
        return self.private_sha256["artifact_sha256_antigravity.json"]

    @property
    def decision_sha256(self) -> str:
        return self.private_sha256["decision_antigravity.json"]

    @property
    def dashboard_snapshot_sha256(self) -> str:
        return self.private_sha256["dashboard_snapshot_antigravity.json"]

    @property
    def summary_sha256(self) -> str:
        return self.private_sha256["method_summary_antigravity.json"]

    @property
    def predecessor_result(self) -> Exp462ResultSnapshotAntigravity:
        return self.result_snapshot


@dataclass(frozen=True, slots=True)
class Exp463MethodFoldMetricAntigravity:
    """Immutable shape for one future method/seed/fold metric row."""

    method: str
    seed: int
    fold_index: int
    normal_trials: int
    normal_alerts: int
    normal_wilson_upper: float
    per_type_auc: Mapping[str, float]
    pooled_auc: float
    scores_finite: bool
    normal_scores: tuple[float, ...]
    primary_scores: Mapping[str, tuple[float, ...]]
    passed: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _method_id_antigravity(self.method)
        _seed_antigravity(self.seed)
        _fold_antigravity(self.fold_index)
        trials = _int_antigravity(self.normal_trials, "normal_trials", positive=True)
        alerts = _int_antigravity(self.normal_alerts, "normal_alerts")
        if alerts > trials:
            raise ValueError("normal_alerts must not exceed normal_trials")
        _unit_float_antigravity(self.normal_wilson_upper, "normal_wilson_upper")
        per_type = _per_type_auc_antigravity(self.per_type_auc, "per_type_auc")
        pooled = _unit_float_antigravity(self.pooled_auc, "pooled_auc")
        if pooled != float(sum(per_type.values()) / len(_CHANGE_TYPES_ANTIGRAVITY)):
            raise ValueError("pooled_auc must equal the four-type mean")
        finite = _bool_antigravity(self.scores_finite, "scores_finite")
        normal_scores = _score_tuple_antigravity(self.normal_scores, "normal_scores")
        if len(normal_scores) != trials:
            raise ValueError("normal_trials must equal normal score evidence length")
        if not isinstance(self.primary_scores, Mapping) or tuple(self.primary_scores) != _CHANGE_TYPES_ANTIGRAVITY:
            raise ValueError("primary_scores must preserve change order")
        primary = {
            key: _score_tuple_antigravity(self.primary_scores[key], f"primary_scores[{key}]")
            for key in _CHANGE_TYPES_ANTIGRAVITY
        }
        if any(len(value) != trials for value in primary.values()):
            raise ValueError("primary score evidence lengths must equal normal trials")
        passed = _bool_antigravity(self.passed, "passed")
        reasons = _reason_tuple_antigravity(self.failure_reasons)
        object.__setattr__(self, "per_type_auc", per_type)
        object.__setattr__(self, "normal_scores", normal_scores)
        object.__setattr__(self, "primary_scores", MappingProxyType(primary))
        if passed and reasons:
            raise ValueError("passed row cannot carry failure reasons")


@dataclass(frozen=True, slots=True)
class Exp463MethodSummaryAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_per_type_auc: Mapping[str, float]
    min_gradual_drift_auc: float
    min_period_change_auc: float
    min_pooled_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool
    passed: bool

    def __post_init__(self) -> None:
        _method_id_antigravity(self.method)
        fold_count = _int_antigravity(self.fold_count, "fold_count", positive=True)
        passed_count = _int_antigravity(self.passed_count, "passed_count")
        failed_count = _int_antigravity(self.failed_count, "failed_count")
        if fold_count != passed_count + failed_count:
            raise ValueError("summary counts must cover all folds")
        per_type = _per_type_auc_antigravity(self.min_per_type_auc, "min_per_type_auc")
        _unit_float_antigravity(self.min_gradual_drift_auc, "min_gradual_drift_auc")
        _unit_float_antigravity(self.min_period_change_auc, "min_period_change_auc")
        _unit_float_antigravity(self.min_pooled_auc, "min_pooled_auc")
        _unit_float_antigravity(self.max_normal_wilson_upper, "max_normal_wilson_upper")
        _bool_antigravity(self.all_scores_finite, "all_scores_finite")
        _bool_antigravity(self.passed, "passed")
        object.__setattr__(self, "min_per_type_auc", per_type)


@dataclass(frozen=True, slots=True)
class Exp463QualificationContractAntigravity:
    """Future Task 3/4 container contract; no metric computation is done."""

    dataset: str
    method_order: tuple[str, ...]
    primary_candidate: str
    fold_metrics: tuple[Exp463MethodFoldMetricAntigravity, ...]
    summaries: tuple[Exp463MethodSummaryAntigravity, ...]
    label_query_count: int

    def __post_init__(self) -> None:
        if self.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
            raise ValueError("dataset differs from frozen TRAIN contract")
        if self.method_order != _METHOD_IDS_ANTIGRAVITY:
            raise ValueError("method order must preserve M0-M6")
        if self.primary_candidate != FROZEN_CONFIG_ANTIGRAVITY.primary_candidate:
            raise ValueError("primary candidate differs")
        if type(self.fold_metrics) is not tuple:
            raise ValueError("fold_metrics must be tuple")
        if len(self.fold_metrics) != self.expected_metric_row_count_antigravity():
            raise ValueError("fold_metrics must contain exactly 70 rows")
        if any(type(row) is not Exp463MethodFoldMetricAntigravity for row in self.fold_metrics):
            raise ValueError("fold_metrics contain an invalid row type")
        expected = tuple(
            (method, seed, fold_index)
            for method in _METHOD_IDS_ANTIGRAVITY
            for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if tuple((row.method, row.seed, row.fold_index) for row in self.fold_metrics) != expected:
            raise ValueError("fold metric order differs from M0-M6/seed/fold order")
        if type(self.summaries) is not tuple or len(self.summaries) != self.expected_summary_count_antigravity():
            raise ValueError("summaries must contain exactly seven methods")
        if tuple(summary.method for summary in self.summaries) != _METHOD_IDS_ANTIGRAVITY:
            raise ValueError("summary order must preserve M0-M6")
        if any(type(summary) is not Exp463MethodSummaryAntigravity for summary in self.summaries):
            raise ValueError("summaries contain an invalid type")
        if type(self.label_query_count) is not int or self.label_query_count != 0:
            raise ValueError("label query count must remain zero")

    @staticmethod
    def expected_metric_row_count_antigravity() -> int:
        return len(_METHOD_IDS_ANTIGRAVITY) * len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds

    @staticmethod
    def expected_summary_count_antigravity() -> int:
        return len(_METHOD_IDS_ANTIGRAVITY)


def validate_exp462_authority_antigravity(
    authority: Exp462AuthorityAntigravity,
) -> Exp462AuthorityAntigravity:
    """Reconstruct the strict frozen authority to reject copied mutations."""

    if type(authority) is not Exp462AuthorityAntigravity:
        raise ValueError("authority type is invalid")
    return Exp462AuthorityAntigravity(
        **{field.name: getattr(authority, field.name) for field in fields(type(authority))}
    )


# Controlled aliases retain the Exp462 naming convention for later Task 4 code
# without creating a second mutable contract implementation.
MethodFoldMetricAntigravity = Exp463MethodFoldMetricAntigravity
MethodSummaryAntigravity = Exp463MethodSummaryAntigravity
MethodQualificationAntigravity = Exp463QualificationContractAntigravity
Exp462HistoricalPeriodFoldAntigravity = Exp463HistoricalPeriodFoldAntigravity
HistoricalPeriodFoldAntigravity = Exp463HistoricalPeriodFoldAntigravity
Exp462QualificationAuthorityAntigravity = Exp462AuthorityAntigravity
Exp462ResultSummaryAntigravity = Exp462MethodSummaryAntigravity
Exp463FoldMetricAntigravity = Exp463MethodFoldMetricAntigravity
Exp463SummaryAntigravity = Exp463MethodSummaryAntigravity
Exp463QualificationAntigravity = Exp463QualificationContractAntigravity


__all__ = (
    "Exp463HistoricalPeriodFoldAntigravity",
    "Exp462HistoricalPeriodFoldAntigravity",
    "HistoricalPeriodFoldAntigravity",
    "Exp462ResultRowIdentityAntigravity",
    "Exp462MethodSummaryAntigravity",
    "Exp462ResultSnapshotAntigravity",
    "Exp462AuthorityAntigravity",
    "Exp462QualificationAuthorityAntigravity",
    "Exp462ResultSummaryAntigravity",
    "Exp463MethodFoldMetricAntigravity",
    "Exp463FoldMetricAntigravity",
    "Exp463MethodSummaryAntigravity",
    "Exp463SummaryAntigravity",
    "Exp463QualificationContractAntigravity",
    "Exp463QualificationAntigravity",
    "MethodFoldMetricAntigravity",
    "MethodSummaryAntigravity",
    "MethodQualificationAntigravity",
    "validate_exp462_authority_antigravity",
)
