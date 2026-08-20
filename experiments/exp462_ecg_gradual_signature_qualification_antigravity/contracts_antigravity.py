"""Immutable Exp462 Task 2 contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
import re
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_PRIVATE_SHA256_ANTIGRAVITY = MappingProxyType(
    {
        "artifact_map_antigravity.json": FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256,
        "decision_antigravity.json": FROZEN_CONFIG_ANTIGRAVITY.predecessor_decision_sha256,
        "dashboard_snapshot_antigravity.json": FROZEN_CONFIG_ANTIGRAVITY.predecessor_dashboard_snapshot_sha256,
        "summary_antigravity.json": FROZEN_CONFIG_ANTIGRAVITY.predecessor_summary_sha256,
    }
)


def _int_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed_antigravity(value: object) -> int:
    if type(value) is not int or value not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the exact seeds")
    return value


def _fold_index_antigravity(value: object) -> int:
    if type(value) is not int or not 0 <= value < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be an exact blocked fold index")
    return value


def _auc_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _sha40_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{name} must be a lowercase 40-hex commit")
    return value


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex sha256")
    return value


def _scalar_signature_antigravity(value: object) -> tuple[type, object]:
    return (type(value), value)


@dataclass(frozen=True, slots=True)
class Exp461HistoricalPeriodFoldAntigravity:
    seed: int
    fold_index: int
    period_change_auc: float

    def __post_init__(self) -> None:
        _seed_antigravity(self.seed)
        _fold_index_antigravity(self.fold_index)
        _auc_antigravity(self.period_change_auc, "period_change_auc")


@dataclass(frozen=True, slots=True)
class Exp461GradualAuthorityAntigravity:
    predecessor_experiment: str
    terminal_commit: str
    artifact_map_sha256: str
    decision_sha256: str
    dashboard_snapshot_sha256: str
    summary_sha256: str
    status: str
    primary_candidate: str
    passed_record_count: int
    required_record_count: int
    failed_gradual_drift_auc: float
    period_min_auc: float
    historical_period_folds: tuple[Exp461HistoricalPeriodFoldAntigravity, ...]
    label_query_count: int
    private_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.predecessor_experiment) is not str:
            raise ValueError("predecessor_experiment must be str")
        _sha40_antigravity(self.terminal_commit, "terminal_commit")
        _sha64_antigravity(self.artifact_map_sha256, "artifact_map_sha256")
        _sha64_antigravity(self.decision_sha256, "decision_sha256")
        _sha64_antigravity(self.dashboard_snapshot_sha256, "dashboard_snapshot_sha256")
        _sha64_antigravity(self.summary_sha256, "summary_sha256")
        if type(self.status) is not str:
            raise ValueError("status must be str")
        if type(self.primary_candidate) is not str:
            raise ValueError("primary_candidate must be str")
        _int_antigravity(self.passed_record_count, "passed_record_count")
        _int_antigravity(self.required_record_count, "required_record_count")
        _auc_antigravity(self.failed_gradual_drift_auc, "failed_gradual_drift_auc")
        _auc_antigravity(self.period_min_auc, "period_min_auc")
        if type(self.historical_period_folds) is not tuple or len(self.historical_period_folds) != 2:
            raise ValueError("historical_period_folds must be the exact two-record tuple")
        if any(type(row) is not Exp461HistoricalPeriodFoldAntigravity for row in self.historical_period_folds):
            raise ValueError("historical_period_folds must contain Exp461HistoricalPeriodFoldAntigravity rows")
        if not isinstance(self.private_sha256, Mapping):
            raise ValueError("private_sha256 must be a mapping")
        if tuple(self.private_sha256) != tuple(_PRIVATE_SHA256_ANTIGRAVITY):
            raise ValueError("private_sha256 must preserve the exact four-hash order")
        frozen_private = {
            key: _sha64_antigravity(self.private_sha256[key], f"private_sha256[{key}]")
            for key in _PRIVATE_SHA256_ANTIGRAVITY
        }
        object.__setattr__(self, "private_sha256", MappingProxyType(frozen_private))
        if _int_antigravity(self.label_query_count, "label_query_count") != 0:
            raise ValueError("label_query_count must remain zero")
        actual_snapshot = {
            "predecessor_experiment": _scalar_signature_antigravity(self.predecessor_experiment),
            "terminal_commit": _scalar_signature_antigravity(self.terminal_commit),
            "artifact_map_sha256": _scalar_signature_antigravity(self.artifact_map_sha256),
            "decision_sha256": _scalar_signature_antigravity(self.decision_sha256),
            "dashboard_snapshot_sha256": _scalar_signature_antigravity(self.dashboard_snapshot_sha256),
            "summary_sha256": _scalar_signature_antigravity(self.summary_sha256),
            "status": _scalar_signature_antigravity(self.status),
            "primary_candidate": _scalar_signature_antigravity(self.primary_candidate),
            "passed_record_count": _scalar_signature_antigravity(self.passed_record_count),
            "required_record_count": _scalar_signature_antigravity(self.required_record_count),
            "failed_gradual_drift_auc": _scalar_signature_antigravity(self.failed_gradual_drift_auc),
            "period_min_auc": _scalar_signature_antigravity(self.period_min_auc),
            "historical_period_folds": tuple(
                (
                    _scalar_signature_antigravity(row.seed),
                    _scalar_signature_antigravity(row.fold_index),
                    _scalar_signature_antigravity(row.period_change_auc),
                )
                for row in self.historical_period_folds
            ),
            "label_query_count": _scalar_signature_antigravity(self.label_query_count),
            "private_sha256": tuple(
                (
                    _scalar_signature_antigravity(key),
                    _scalar_signature_antigravity(self.private_sha256[key]),
                )
                for key in self.private_sha256
            ),
        }
        frozen_snapshot = {
            "predecessor_experiment": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment),
            "terminal_commit": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_commit),
            "artifact_map_sha256": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256),
            "decision_sha256": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_decision_sha256),
            "dashboard_snapshot_sha256": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_dashboard_snapshot_sha256),
            "summary_sha256": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_summary_sha256),
            "status": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_status),
            "primary_candidate": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate),
            "passed_record_count": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count),
            "required_record_count": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count),
            "failed_gradual_drift_auc": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc),
            "period_min_auc": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.predecessor_period_min_auc),
            "historical_period_folds": (
                (
                    _scalar_signature_antigravity(20260721),
                    _scalar_signature_antigravity(3),
                    _scalar_signature_antigravity(0.907392736783864),
                ),
                (
                    _scalar_signature_antigravity(20261721),
                    _scalar_signature_antigravity(4),
                    _scalar_signature_antigravity(0.9216148584075248),
                ),
            ),
            "label_query_count": _scalar_signature_antigravity(FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain),
            "private_sha256": tuple(
                (
                    _scalar_signature_antigravity(key),
                    _scalar_signature_antigravity(value),
                )
                for key, value in _PRIVATE_SHA256_ANTIGRAVITY.items()
            ),
        }
        if actual_snapshot != frozen_snapshot:
            raise ValueError("Exp461 authority differs from the exact frozen snapshot")


def validate_exp461_gradual_authority_antigravity(
    authority: Exp461GradualAuthorityAntigravity,
) -> Exp461GradualAuthorityAntigravity:
    if type(authority) is not Exp461GradualAuthorityAntigravity:
        raise ValueError("authority type is invalid")
    return Exp461GradualAuthorityAntigravity(
        **{field.name: getattr(authority, field.name) for field in fields(type(authority))}
    )


__all__ = (
    "Exp461HistoricalPeriodFoldAntigravity",
    "Exp461GradualAuthorityAntigravity",
    "validate_exp461_gradual_authority_antigravity",
)
