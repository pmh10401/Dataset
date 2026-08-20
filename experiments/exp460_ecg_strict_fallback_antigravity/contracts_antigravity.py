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


@dataclass(frozen=True, slots=True)
class FallbackRuleComparisonAntigravity:
    rule: str
    route: str
    tp: int
    fp: int
    fn: int
    tn: int
    f1: float

    def __post_init__(self) -> None:
        if type(self.rule) is not str or self.rule not in FROZEN_CONFIG_ANTIGRAVITY.rule_order:
            raise ValueError("rule must be an exact Exp460 fallback rule")
        if type(self.route) is not str or self.route not in {"baseline", "candidate"}:
            raise ValueError("route must be baseline or candidate")
        for name in ("tp", "fp", "fn", "tn"):
            _int_antigravity(getattr(self, name), name)
        _auc_antigravity(self.f1, "f1")


@dataclass(frozen=True, slots=True)
class FallbackEvaluationAntigravity:
    rule_results: tuple[
        FallbackRuleComparisonAntigravity,
        FallbackRuleComparisonAntigravity,
        FallbackRuleComparisonAntigravity,
        FallbackRuleComparisonAntigravity,
    ]
    strict_route: str
    strict_status: str
    joint_success: bool
    all_records_passed: bool
    passed_record_count: int
    worst_period_change_auc: float

    def __post_init__(self) -> None:
        if type(self.rule_results) is not tuple or len(self.rule_results) != len(
            FROZEN_CONFIG_ANTIGRAVITY.rule_order
        ):
            raise ValueError("rule_results must be the exact four-row tuple")
        if any(type(row) is not FallbackRuleComparisonAntigravity for row in self.rule_results):
            raise ValueError("rule_results must contain FallbackRuleComparisonAntigravity rows")
        if type(self.strict_route) is not str or self.strict_route not in {"baseline", "candidate"}:
            raise ValueError("strict_route must be baseline or candidate")
        if type(self.strict_status) is not str or self.strict_status not in {
            "candidate_retained_joint_success",
            "candidate_retained_performance_not_improved",
            "fallback_triggered_performance_not_improved",
        }:
            raise ValueError("strict_status must be an exact Exp460 Task 3 status")
        _bool_antigravity(self.joint_success, "joint_success")
        _bool_antigravity(self.all_records_passed, "all_records_passed")
        passed_record_count = _int_antigravity(self.passed_record_count, "passed_record_count")
        if passed_record_count > len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("passed_record_count exceeds the ten frozen records")
        _auc_antigravity(self.worst_period_change_auc, "worst_period_change_auc")


def _task3_snapshot_from_authority_antigravity(
    authority: object,
) -> dict[str, object]:
    if type(authority) is not Exp459FallbackAuthorityAntigravity:
        raise ValueError("Exp459FallbackAuthorityAntigravity is required")

    fold_records = authority.fold_records
    if type(fold_records) is not tuple or len(fold_records) != (
        len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
    ):
        raise ValueError("fold_records must be the exact ten-record tuple")
    expected_order = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    actual_order: list[tuple[int, int]] = []
    passes: list[bool] = []
    aucs: list[float] = []
    for record in fold_records:
        if type(record) is not Exp459FoldRecordAntigravity:
            raise ValueError("fold_records must contain Exp459FoldRecordAntigravity records")
        actual_order.append((_seed_antigravity(record.seed), _fold_index_antigravity(record.fold_index)))
        aucs.append(_auc_antigravity(record.period_change_auc, "period_change_auc"))
        passes.append(_bool_antigravity(record.passed, "passed"))
    if tuple(actual_order) != expected_order:
        raise ValueError("fold_records must stay in exact seed-then-fold order")

    seed_summaries = authority.seed_summaries
    if type(seed_summaries) is not tuple or len(seed_summaries) != len(
        FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
    ):
        raise ValueError("seed_summaries must be the exact two-record tuple")
    if tuple(summary.seed for summary in seed_summaries) != FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds:
        raise ValueError("seed_summaries must stay in exact Hydra seed order")
    for index, summary in enumerate(seed_summaries):
        if type(summary) is not Exp459SeedSummaryAntigravity:
            raise ValueError("seed_summaries must contain Exp459SeedSummaryAntigravity records")
        _seed_antigravity(summary.seed)
        expected_slice = fold_records[
            index * FROZEN_CONFIG_ANTIGRAVITY.folds : (index + 1) * FROZEN_CONFIG_ANTIGRAVITY.folds
        ]
        if type(summary.fold_records) is not tuple or summary.fold_records != expected_slice:
            raise ValueError("seed summary fold_records differ from authority fold inventory")
        if _bool_antigravity(summary.passed, "passed") is not all(
            record.passed for record in summary.fold_records
        ):
            raise ValueError("seed summary passed differs from fold pass recomputation")

    if type(authority.decision) is not Exp459DecisionAntigravity:
        raise ValueError("decision type is invalid")
    all_records_passed = all(passes)
    if _bool_antigravity(authority.decision.passed, "passed") is not all_records_passed:
        raise ValueError("decision differs from fold pass recomputation")
    if type(authority.decision.status) is not str or authority.decision.status not in {
        "stable",
        "unstable",
    }:
        raise ValueError("status must be stable or unstable")
    if authority.decision.status != ("stable" if all_records_passed else "unstable"):
        raise ValueError("decision status differs from fold pass recomputation")

    if type(authority.baseline_confusion) is not ConfusionAntigravity:
        raise ValueError("baseline_confusion type is invalid")
    if type(authority.candidate_confusion) is not ConfusionAntigravity:
        raise ValueError("candidate_confusion type is invalid")
    baseline_confusion = authority.baseline_confusion
    candidate_confusion = authority.candidate_confusion
    for name in ("tp", "fp", "fn", "tn"):
        _int_antigravity(getattr(baseline_confusion, name), f"baseline_confusion.{name}")
        _int_antigravity(getattr(candidate_confusion, name), f"candidate_confusion.{name}")

    _sha_antigravity(authority.frozen_authority_digest, "frozen_authority_digest")
    _sha_antigravity(authority.official_artifact_map_sha256, "official_artifact_map_sha256")
    _sha_antigravity(authority.decision_sha256, "decision_sha256")
    if _int_antigravity(authority.label_query_count, "label_query_count") != 0:
        raise ValueError("label_query_count must remain zero")
    return {
        "passed_record_count": sum(1 for passed in passes if passed),
        "all_records_passed": all_records_passed,
        "worst_period_change_auc": min(aucs),
        "baseline_confusion": baseline_confusion,
        "candidate_confusion": candidate_confusion,
    }


def _f1_antigravity(tp: int, fp: int, fn: int) -> float:
    denominator = (2 * tp) + fp + fn
    if denominator <= 0:
        raise ValueError("f1 denominator must be positive")
    return (2 * tp) / denominator


def _status_antigravity(strict_route: str, joint_success: bool) -> str:
    if strict_route == "baseline":
        return "fallback_triggered_performance_not_improved"
    if joint_success:
        return "candidate_retained_joint_success"
    return "candidate_retained_performance_not_improved"


def validate_fallback_evaluation_antigravity(
    value: object,
    authority: object,
) -> FallbackEvaluationAntigravity:
    if type(value) is not FallbackEvaluationAntigravity:
        raise ValueError("FallbackEvaluationAntigravity is required")
    snapshot = _task3_snapshot_from_authority_antigravity(authority)
    baseline_confusion = snapshot["baseline_confusion"]
    candidate_confusion = snapshot["candidate_confusion"]
    assert type(baseline_confusion) is ConfusionAntigravity
    assert type(candidate_confusion) is ConfusionAntigravity
    passed_record_count = snapshot["passed_record_count"]
    all_records_passed = snapshot["all_records_passed"]
    worst_period_change_auc = snapshot["worst_period_change_auc"]
    assert type(passed_record_count) is int
    assert type(all_records_passed) is bool
    assert type(worst_period_change_auc) is float

    if tuple(row.rule for row in value.rule_results) != FROZEN_CONFIG_ANTIGRAVITY.rule_order:
        raise ValueError("rule_results must stay in exact Exp460 rule order")
    expected_routes = (
        "baseline",
        "candidate",
        "candidate" if passed_record_count >= 8 else "baseline",
        "candidate" if all_records_passed else "baseline",
    )
    baseline_row = value.rule_results[0]
    strict_row = value.rule_results[3]
    for row, expected_route in zip(value.rule_results, expected_routes, strict=True):
        if row.route != expected_route:
            raise ValueError("row route differs from recomputed authority route")
        expected_confusion = baseline_confusion if expected_route == "baseline" else candidate_confusion
        if (row.tp, row.fp, row.fn, row.tn) != (
            expected_confusion.tp,
            expected_confusion.fp,
            expected_confusion.fn,
            expected_confusion.tn,
        ):
            raise ValueError("row confusion differs from the selected route confusion")
        if row.f1 != _f1_antigravity(row.tp, row.fp, row.fn):
            raise ValueError("row f1 differs from confusion recomputation")
    if value.strict_route != strict_row.route:
        raise ValueError("strict_route differs from the strict rule row")
    if value.all_records_passed is not all_records_passed:
        raise ValueError("all_records_passed differs from fold pass recomputation")
    if value.passed_record_count != passed_record_count:
        raise ValueError("passed_record_count differs from fold pass recomputation")
    if value.worst_period_change_auc != worst_period_change_auc:
        raise ValueError("worst_period_change_auc differs from fold AUC recomputation")

    joint_success = (
        value.strict_route == "candidate"
        and value.all_records_passed
        and value.worst_period_change_auc >= FROZEN_CONFIG_ANTIGRAVITY.period_change_auc_min
        and strict_row.fn < baseline_row.fn
        and strict_row.fp <= baseline_row.fp
        and strict_row.f1 > baseline_row.f1
    )
    if value.joint_success is not joint_success:
        raise ValueError("joint_success differs from the strict success gate")
    if value.strict_status != _status_antigravity(value.strict_route, value.joint_success):
        raise ValueError("strict_status differs from the strict success gate")
    return FallbackEvaluationAntigravity(
        rule_results=value.rule_results,
        strict_route=value.strict_route,
        strict_status=value.strict_status,
        joint_success=value.joint_success,
        all_records_passed=value.all_records_passed,
        passed_record_count=value.passed_record_count,
        worst_period_change_auc=value.worst_period_change_auc,
    )


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
    "FallbackEvaluationAntigravity",
    "FallbackRuleComparisonAntigravity",
    "validate_exp459_fallback_authority_antigravity",
    "validate_fallback_evaluation_antigravity",
)
