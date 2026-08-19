"""Immutable validated Task 2 predecessor contracts for Exp460."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
    EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
)
from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
    FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


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
        raise ValueError("seed must be one of the exact Exp459 Hydra seeds")
    return value


def _fold_index_antigravity(value: object) -> int:
    if type(value) is not int or not 0 <= value < FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("fold_index must be an exact Exp459 fold index")
    return value


def _auc_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


def _sha_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex sha256")
    return value


def _scalar_signature_antigravity(value: object) -> tuple[type, object]:
    return (type(value), value)


def _expected_fold_rows_antigravity() -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            _scalar_signature_antigravity(record.seed),
            _scalar_signature_antigravity(record.fold_index),
            _scalar_signature_antigravity(record.per_type_auc["period_change"]),
            _scalar_signature_antigravity(record.passed),
        )
        for record in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence
    )


def _expected_seed_rows_antigravity() -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            _scalar_signature_antigravity(summary.seed),
            tuple(
                (
                    _scalar_signature_antigravity(record.seed),
                    _scalar_signature_antigravity(record.fold_index),
                    _scalar_signature_antigravity(record.per_type_auc["period_change"]),
                    _scalar_signature_antigravity(record.passed),
                )
                for record in summary.fold_records
            ),
            _scalar_signature_antigravity(summary.passed),
        )
        for summary in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.seed_summaries
    )


def _expected_snapshot_antigravity() -> dict[str, object]:
    return {
        "fold_records": _expected_fold_rows_antigravity(),
        "seed_summaries": _expected_seed_rows_antigravity(),
        "decision": (
            _scalar_signature_antigravity(FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.decision.status),
            _scalar_signature_antigravity(FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.decision.passed),
        ),
        "baseline_confusion": tuple(
            _scalar_signature_antigravity(value)
            for value in FROZEN_CONFIG_ANTIGRAVITY.baseline_confusion
        ),
        "candidate_confusion": tuple(
            _scalar_signature_antigravity(value)
            for value in (
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tp,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fp,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fn,
                EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tn,
            )
        ),
        "frozen_authority_digest": _scalar_signature_antigravity(
            FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.canonical_digest
        ),
        "official_artifact_map_sha256": _scalar_signature_antigravity(
            FROZEN_CONFIG_ANTIGRAVITY.exp459_official_artifact_map_sha256
        ),
        "decision_sha256": _scalar_signature_antigravity(
            FROZEN_CONFIG_ANTIGRAVITY.exp459_decision_sha256
        ),
        "label_query_count": _scalar_signature_antigravity(
            FROZEN_CONFIG_ANTIGRAVITY.label_query_count
        ),
    }


@dataclass(frozen=True, slots=True)
class ConfusionAntigravity:
    tp: int
    fp: int
    fn: int
    tn: int

    def __post_init__(self) -> None:
        for name in ("tp", "fp", "fn", "tn"):
            _int_antigravity(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class Exp459FoldRecordAntigravity:
    seed: int
    fold_index: int
    period_change_auc: float
    passed: bool

    def __post_init__(self) -> None:
        _seed_antigravity(self.seed)
        _fold_index_antigravity(self.fold_index)
        _auc_antigravity(self.period_change_auc, "period_change_auc")
        _bool_antigravity(self.passed, "passed")


@dataclass(frozen=True, slots=True)
class Exp459SeedSummaryAntigravity:
    seed: int
    fold_records: tuple[Exp459FoldRecordAntigravity, ...]
    passed: bool

    def __post_init__(self) -> None:
        seed = _seed_antigravity(self.seed)
        if type(self.fold_records) is not tuple or len(self.fold_records) != FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_records must be the exact five-record tuple")
        if any(type(record) is not Exp459FoldRecordAntigravity for record in self.fold_records):
            raise ValueError("fold_records must contain Exp459FoldRecordAntigravity records")
        if any(record.seed != seed for record in self.fold_records):
            raise ValueError("fold_records must match the summary seed")
        if tuple(record.fold_index for record in self.fold_records) != tuple(
            range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        ):
            raise ValueError("fold_records must remain in exact fold order")
        if _bool_antigravity(self.passed, "passed") is not all(
            record.passed for record in self.fold_records
        ):
            raise ValueError("seed summary passed differs from fold pass recomputation")


@dataclass(frozen=True, slots=True)
class Exp459DecisionAntigravity:
    status: str
    passed: bool

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in {"stable", "unstable"}:
            raise ValueError("status must be stable or unstable")
        passed = _bool_antigravity(self.passed, "passed")
        if passed is not (self.status == "stable"):
            raise ValueError("decision passed differs from status")


@dataclass(frozen=True, slots=True)
class Exp459FallbackAuthorityAntigravity:
    fold_records: tuple[Exp459FoldRecordAntigravity, ...]
    seed_summaries: tuple[Exp459SeedSummaryAntigravity, Exp459SeedSummaryAntigravity]
    decision: Exp459DecisionAntigravity
    baseline_confusion: ConfusionAntigravity
    candidate_confusion: ConfusionAntigravity
    frozen_authority_digest: str
    official_artifact_map_sha256: str
    decision_sha256: str
    label_query_count: int

    def __post_init__(self) -> None:
        if type(self.fold_records) is not tuple or len(self.fold_records) != (
            len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_records must be the exact ten-record tuple")
        if any(type(record) is not Exp459FoldRecordAntigravity for record in self.fold_records):
            raise ValueError("fold_records must contain Exp459FoldRecordAntigravity records")
        expected_order = tuple(
            (seed, fold_index)
            for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
            for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        if tuple((record.seed, record.fold_index) for record in self.fold_records) != expected_order:
            raise ValueError("fold_records must stay in exact seed-then-fold order")
        if type(self.seed_summaries) is not tuple or len(self.seed_summaries) != len(
            FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        ):
            raise ValueError("seed_summaries must be the exact two-record tuple")
        if any(type(summary) is not Exp459SeedSummaryAntigravity for summary in self.seed_summaries):
            raise ValueError("seed_summaries must contain Exp459SeedSummaryAntigravity records")
        if tuple(summary.seed for summary in self.seed_summaries) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
            raise ValueError("seed_summaries must stay in exact Hydra seed order")
        for index, summary in enumerate(self.seed_summaries):
            start = index * FROZEN_CONFIG_ANTIGRAVITY.folds
            stop = start + FROZEN_CONFIG_ANTIGRAVITY.folds
            if summary.fold_records != self.fold_records[start:stop]:
                raise ValueError("seed summary fold_records differ from authority fold inventory")
        if type(self.decision) is not Exp459DecisionAntigravity:
            raise ValueError("decision type is invalid")
        if type(self.baseline_confusion) is not ConfusionAntigravity:
            raise ValueError("baseline_confusion type is invalid")
        if type(self.candidate_confusion) is not ConfusionAntigravity:
            raise ValueError("candidate_confusion type is invalid")
        _sha_antigravity(self.frozen_authority_digest, "frozen_authority_digest")
        _sha_antigravity(
            self.official_artifact_map_sha256,
            "official_artifact_map_sha256",
        )
        _sha_antigravity(self.decision_sha256, "decision_sha256")
        if type(self.label_query_count) is not int:
            raise ValueError("label_query_count must remain int")
        if self.label_query_count != 0:
            raise ValueError("label_query_count must remain zero")
        if self.decision.passed is not all(record.passed for record in self.fold_records):
            raise ValueError("decision differs from fold pass recomputation")
        if self.decision.status != ("stable" if self.decision.passed else "unstable"):
            raise ValueError("decision status differs from fold pass recomputation")
        snapshot = {
            "fold_records": tuple(
                (
                    _scalar_signature_antigravity(record.seed),
                    _scalar_signature_antigravity(record.fold_index),
                    _scalar_signature_antigravity(record.period_change_auc),
                    _scalar_signature_antigravity(record.passed),
                )
                for record in self.fold_records
            ),
            "seed_summaries": tuple(
                (
                    _scalar_signature_antigravity(summary.seed),
                    tuple(
                        (
                            _scalar_signature_antigravity(record.seed),
                            _scalar_signature_antigravity(record.fold_index),
                            _scalar_signature_antigravity(record.period_change_auc),
                            _scalar_signature_antigravity(record.passed),
                        )
                        for record in summary.fold_records
                    ),
                    _scalar_signature_antigravity(summary.passed),
                )
                for summary in self.seed_summaries
            ),
            "decision": (
                _scalar_signature_antigravity(self.decision.status),
                _scalar_signature_antigravity(self.decision.passed),
            ),
            "baseline_confusion": tuple(
                _scalar_signature_antigravity(value)
                for value in (
                    self.baseline_confusion.tp,
                    self.baseline_confusion.fp,
                    self.baseline_confusion.fn,
                    self.baseline_confusion.tn,
                )
            ),
            "candidate_confusion": tuple(
                _scalar_signature_antigravity(value)
                for value in (
                    self.candidate_confusion.tp,
                    self.candidate_confusion.fp,
                    self.candidate_confusion.fn,
                    self.candidate_confusion.tn,
                )
            ),
            "frozen_authority_digest": _scalar_signature_antigravity(
                self.frozen_authority_digest
            ),
            "official_artifact_map_sha256": _scalar_signature_antigravity(
                self.official_artifact_map_sha256
            ),
            "decision_sha256": _scalar_signature_antigravity(self.decision_sha256),
            "label_query_count": _scalar_signature_antigravity(self.label_query_count),
        }
        if snapshot != _expected_snapshot_antigravity():
            raise ValueError("Exp459 fallback authority differs from the exact frozen snapshot")


def validate_exp459_fallback_authority_antigravity(
    value: object,
) -> Exp459FallbackAuthorityAntigravity:
    if type(value) is not Exp459FallbackAuthorityAntigravity:
        raise ValueError("Exp459FallbackAuthorityAntigravity is required")
    return Exp459FallbackAuthorityAntigravity(
        fold_records=value.fold_records,
        seed_summaries=value.seed_summaries,
        decision=value.decision,
        baseline_confusion=value.baseline_confusion,
        candidate_confusion=value.candidate_confusion,
        frozen_authority_digest=value.frozen_authority_digest,
        official_artifact_map_sha256=value.official_artifact_map_sha256,
        decision_sha256=value.decision_sha256,
        label_query_count=value.label_query_count,
    )


__all__ = (
    "ConfusionAntigravity",
    "Exp459DecisionAntigravity",
    "Exp459FallbackAuthorityAntigravity",
    "Exp459FoldRecordAntigravity",
    "Exp459SeedSummaryAntigravity",
    "validate_exp459_fallback_authority_antigravity",
)
