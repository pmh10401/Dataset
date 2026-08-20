"""Immutable Exp461 Task 2 predecessor contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
import re
from types import MappingProxyType

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
    FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_REVIEWED_COMMIT_ANTIGRAVITY = "ca51813450f26b8a3fada6f39cf810452467297e"
_LEAF_SHA256_ANTIGRAVITY = MappingProxyType(
    {
        "context_antigravity.json": "1c30f92d60ca76214f680c4452b1a4418764da84fe5ec1f8a5156f743b84db9f",
        "rule_comparisons_antigravity.jsonl": "3ed3f05f5359cbe4d1a0dbcad371bb3ed951679c1b153d59ff591933fc13dc75",
        "failed_folds_antigravity.jsonl": "880d67769d8e4e40b3b504a0573878bdb31a364a33fdd48d70af0fe2b2a7b198",
        "decision_antigravity.json": "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1",
        "dashboard_snapshot_antigravity.json": "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5",
        "artifact_sha256_antigravity.json": "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
    }
)


def _bool_antigravity(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be bool")
    return value


def _int_antigravity(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _seed_antigravity(value: object) -> int:
    if type(value) is not int or value not in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed must be one of the exact Hydra seeds")
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


def _expected_failed_rows_antigravity() -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            _scalar_signature_antigravity(record.seed),
            _scalar_signature_antigravity(record.fold_index),
            _scalar_signature_antigravity(record.per_type_auc["period_change"]),
        )
        for record in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence
        if not record.passed
    )


@dataclass(frozen=True, slots=True)
class Exp460FailedFoldAntigravity:
    seed: int
    fold_index: int
    period_change_auc: float

    def __post_init__(self) -> None:
        _seed_antigravity(self.seed)
        _fold_index_antigravity(self.fold_index)
        _auc_antigravity(self.period_change_auc, "period_change_auc")


@dataclass(frozen=True, slots=True)
class Exp460PeriodAuthorityAntigravity:
    predecessor_experiment: str
    truth_commit: str
    run_root: str
    reviewed_commit: str
    artifact_map_sha256: str
    dashboard_snapshot_sha256: str
    decision_sha256: str
    status: str
    strict_route: str
    joint_success: bool
    passed_record_count: int
    failed_fold_count: int
    worst_period_change_auc: float
    failed_folds: tuple[Exp460FailedFoldAntigravity, ...]
    label_query_count: int
    leaf_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.predecessor_experiment) is not str:
            raise ValueError("predecessor_experiment must be str")
        _sha40_antigravity(self.truth_commit, "truth_commit")
        if type(self.run_root) is not str:
            raise ValueError("run_root must be str")
        _sha40_antigravity(self.reviewed_commit, "reviewed_commit")
        _sha64_antigravity(self.artifact_map_sha256, "artifact_map_sha256")
        _sha64_antigravity(
            self.dashboard_snapshot_sha256,
            "dashboard_snapshot_sha256",
        )
        _sha64_antigravity(self.decision_sha256, "decision_sha256")
        if type(self.status) is not str:
            raise ValueError("status must be str")
        if type(self.strict_route) is not str:
            raise ValueError("strict_route must be str")
        _bool_antigravity(self.joint_success, "joint_success")
        _int_antigravity(self.passed_record_count, "passed_record_count")
        _int_antigravity(self.failed_fold_count, "failed_fold_count")
        _auc_antigravity(self.worst_period_change_auc, "worst_period_change_auc")
        if type(self.failed_folds) is not tuple or len(self.failed_folds) != 2:
            raise ValueError("failed_folds must be the exact two-record tuple")
        if any(type(row) is not Exp460FailedFoldAntigravity for row in self.failed_folds):
            raise ValueError("failed_folds must contain Exp460FailedFoldAntigravity rows")
        if not isinstance(self.leaf_sha256, Mapping):
            raise ValueError("leaf_sha256 must be a mapping")
        if tuple(self.leaf_sha256) != tuple(_LEAF_SHA256_ANTIGRAVITY):
            raise ValueError("leaf_sha256 must preserve the exact six-leaf order")
        frozen_leaf_sha256 = {
            key: _sha64_antigravity(self.leaf_sha256[key], f"leaf_sha256[{key}]")
            for key in _LEAF_SHA256_ANTIGRAVITY
        }
        object.__setattr__(self, "leaf_sha256", MappingProxyType(frozen_leaf_sha256))
        if self.failed_fold_count != len(self.failed_folds):
            raise ValueError("failed_fold_count must match failed_folds")
        if self.passed_record_count + self.failed_fold_count != (
            len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("passed and failed counts must cover the exact ten records")
        if self.worst_period_change_auc != min(row.period_change_auc for row in self.failed_folds):
            raise ValueError("worst_period_change_auc must equal the minimum failed-fold AUC")
        if _int_antigravity(self.label_query_count, "label_query_count") != 0:
            raise ValueError("label_query_count must remain zero")
        actual_snapshot = {
            "predecessor_experiment": _scalar_signature_antigravity(
                self.predecessor_experiment
            ),
            "truth_commit": _scalar_signature_antigravity(self.truth_commit),
            "run_root": _scalar_signature_antigravity(self.run_root),
            "reviewed_commit": _scalar_signature_antigravity(self.reviewed_commit),
            "artifact_map_sha256": _scalar_signature_antigravity(
                self.artifact_map_sha256
            ),
            "dashboard_snapshot_sha256": _scalar_signature_antigravity(
                self.dashboard_snapshot_sha256
            ),
            "decision_sha256": _scalar_signature_antigravity(self.decision_sha256),
            "status": _scalar_signature_antigravity(self.status),
            "strict_route": _scalar_signature_antigravity(self.strict_route),
            "joint_success": _scalar_signature_antigravity(self.joint_success),
            "passed_record_count": _scalar_signature_antigravity(
                self.passed_record_count
            ),
            "failed_fold_count": _scalar_signature_antigravity(
                self.failed_fold_count
            ),
            "worst_period_change_auc": _scalar_signature_antigravity(
                self.worst_period_change_auc
            ),
            "failed_folds": tuple(
                (
                    _scalar_signature_antigravity(row.seed),
                    _scalar_signature_antigravity(row.fold_index),
                    _scalar_signature_antigravity(row.period_change_auc),
                )
                for row in self.failed_folds
            ),
            "label_query_count": _scalar_signature_antigravity(self.label_query_count),
            "leaf_sha256": tuple(
                (
                    _scalar_signature_antigravity(key),
                    _scalar_signature_antigravity(self.leaf_sha256[key]),
                )
                for key in self.leaf_sha256
            ),
        }
        frozen_snapshot = {
            "predecessor_experiment": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment
            ),
            "truth_commit": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_truth_commit
            ),
            "run_root": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_run_root
            ),
            "reviewed_commit": _scalar_signature_antigravity(_REVIEWED_COMMIT_ANTIGRAVITY),
            "artifact_map_sha256": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256
            ),
            "dashboard_snapshot_sha256": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_dashboard_snapshot_sha256
            ),
            "decision_sha256": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_decision_sha256
            ),
            "status": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_status
            ),
            "strict_route": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_strict_route
            ),
            "joint_success": _scalar_signature_antigravity(False),
            "passed_record_count": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count
            ),
            "failed_fold_count": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_fold_count
            ),
            "worst_period_change_auc": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.predecessor_worst_period_change_auc
            ),
            "failed_folds": _expected_failed_rows_antigravity(),
            "label_query_count": _scalar_signature_antigravity(
                FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain
            ),
            "leaf_sha256": tuple(
                (
                    _scalar_signature_antigravity(key),
                    _scalar_signature_antigravity(value),
                )
                for key, value in _LEAF_SHA256_ANTIGRAVITY.items()
            ),
        }
        if actual_snapshot != frozen_snapshot:
            raise ValueError("Exp460 authority differs from the exact frozen snapshot")


def validate_exp460_period_authority_antigravity(
    authority: Exp460PeriodAuthorityAntigravity,
) -> Exp460PeriodAuthorityAntigravity:
    if type(authority) is not Exp460PeriodAuthorityAntigravity:
        raise ValueError("authority type is invalid")
    return Exp460PeriodAuthorityAntigravity(
        **{field.name: getattr(authority, field.name) for field in fields(type(authority))}
    )


__all__ = (
    "Exp460FailedFoldAntigravity",
    "Exp460PeriodAuthorityAntigravity",
    "validate_exp460_period_authority_antigravity",
)
