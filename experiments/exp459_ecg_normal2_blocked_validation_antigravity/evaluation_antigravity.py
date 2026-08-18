"""Stable/unstable evaluation for Exp459 blocked validation."""

from __future__ import annotations

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BlockedQualificationAntigravity,
    FoldEvidenceAntigravity,
    SeedSummaryAntigravity,
    ValidationDecisionAntigravity,
)


EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY = {
    "experiment": "Exp458",
    "tp": 617,
    "fp": 66,
    "fn": 829,
    "tn": 70788,
    "f1": 0.5796148426491311,
}
_DECISION_EVIDENCE_BY_ID_ANTIGRAVITY: dict[
    int, tuple[ValidationDecisionAntigravity, BlockedQualificationAntigravity]
] = {}


def _scalar_signature_antigravity(value: object) -> tuple[type, object]:
    return (type(value), value)


def _fold_signature_antigravity(record: FoldEvidenceAntigravity) -> tuple[object, ...]:
    return (
        _scalar_signature_antigravity(record.dataset),
        _scalar_signature_antigravity(record.seed),
        _scalar_signature_antigravity(record.fold_index),
        record.fit_indices.tobytes(),
        record.query_indices.tobytes(),
        _scalar_signature_antigravity(record.normal_trials),
        _scalar_signature_antigravity(record.normal_alerts),
        _scalar_signature_antigravity(record.normal_wilson_upper),
        tuple(
            _scalar_signature_antigravity(record.per_type_auc[name])
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        ),
        _scalar_signature_antigravity(record.pooled_auc),
        _scalar_signature_antigravity(record.passed),
    )


def _summary_signature_antigravity(summary: SeedSummaryAntigravity) -> tuple[object, ...]:
    return (
        _scalar_signature_antigravity(summary.dataset),
        _scalar_signature_antigravity(summary.seed),
        tuple(_fold_signature_antigravity(record) for record in summary.fold_records),
        _scalar_signature_antigravity(summary.passed),
    )


def _qualification_signature_antigravity(
    qualification: BlockedQualificationAntigravity,
) -> tuple[object, ...]:
    return (
        _scalar_signature_antigravity(qualification.dataset),
        tuple(_fold_signature_antigravity(record) for record in qualification.fold_evidence),
        tuple(_summary_signature_antigravity(summary) for summary in qualification.seed_summaries),
        _scalar_signature_antigravity(qualification.passed),
    )


def _decision_signature_antigravity(decision: ValidationDecisionAntigravity) -> tuple[object, ...]:
    return (
        _scalar_signature_antigravity(decision.dataset),
        _qualification_signature_antigravity(decision.qualification),
        _scalar_signature_antigravity(decision.status),
        _scalar_signature_antigravity(decision.passed),
    )


def _rebuilt_fold_antigravity(record: object) -> FoldEvidenceAntigravity:
    if type(record) is not FoldEvidenceAntigravity:
        raise ValueError("fold evidence type is invalid")
    if (
        type(record.fit_indices) is not np.ndarray
        or record.fit_indices.dtype != np.dtype(np.int64)
        or record.fit_indices.flags.writeable
        or type(record.query_indices) is not np.ndarray
        or record.query_indices.dtype != np.dtype(np.int64)
        or record.query_indices.flags.writeable
    ):
        raise ValueError("fold index arrays must remain exact immutable int64 arrays")
    if tuple(record.per_type_auc) != FROZEN_CONFIG_ANTIGRAVITY.primary_types:
        raise ValueError("per_type_auc must retain exact primary type order")
    return FoldEvidenceAntigravity(
        dataset=record.dataset,
        seed=record.seed,
        fold_index=record.fold_index,
        fit_indices=record.fit_indices,
        query_indices=record.query_indices,
        normal_trials=record.normal_trials,
        normal_alerts=record.normal_alerts,
        normal_wilson_upper=record.normal_wilson_upper,
        per_type_auc={
            name: record.per_type_auc[name]
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        },
        pooled_auc=record.pooled_auc,
        passed=record.passed,
    )


def validate_blocked_qualification_semantics_antigravity(
    value: object,
) -> BlockedQualificationAntigravity:
    if type(value) is not BlockedQualificationAntigravity:
        raise ValueError("qualification type is invalid")
    if value.dataset != FROZEN_CONFIG_ANTIGRAVITY.dataset:
        raise ValueError("qualification dataset differs")
    if type(value.fold_evidence) is not tuple or len(value.fold_evidence) != (
        len(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds) * FROZEN_CONFIG_ANTIGRAVITY.folds
    ):
        raise ValueError("fold evidence inventory differs")
    if type(value.passed) is not bool:
        raise ValueError("qualification passed must remain bool")
    rebuilt_records = tuple(_rebuilt_fold_antigravity(record) for record in value.fold_evidence)
    expected_order = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if tuple((record.seed, record.fold_index) for record in rebuilt_records) != expected_order:
        raise ValueError("qualification fold evidence order differs")

    rebuilt_summaries: list[SeedSummaryAntigravity] = []
    if type(value.seed_summaries) is not tuple or len(value.seed_summaries) != len(
        FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
    ):
        raise ValueError("seed summaries inventory differs")
    for summary_index, summary in enumerate(value.seed_summaries):
        if type(summary) is not SeedSummaryAntigravity:
            raise ValueError("seed summary type is invalid")
        seed = FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds[summary_index]
        if summary.seed != seed:
            raise ValueError("seed summary order differs")
        start = summary_index * FROZEN_CONFIG_ANTIGRAVITY.folds
        stop = start + FROZEN_CONFIG_ANTIGRAVITY.folds
        rebuilt = SeedSummaryAntigravity(
            dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
            seed=seed,
            fold_records=rebuilt_records[start:stop],
            passed=all(record.passed for record in rebuilt_records[start:stop]),
        )
        if _summary_signature_antigravity(summary) != _summary_signature_antigravity(rebuilt):
            raise ValueError("seed summary differs from fold recomputation")
        rebuilt_summaries.append(rebuilt)

    rebuilt_qualification = BlockedQualificationAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        fold_evidence=rebuilt_records,
        seed_summaries=tuple(rebuilt_summaries),  # type: ignore[arg-type]
        passed=all(summary.passed for summary in rebuilt_summaries),
    )
    if _qualification_signature_antigravity(value) != _qualification_signature_antigravity(
        rebuilt_qualification
    ):
        raise ValueError("qualification differs from semantic recomputation")
    return rebuilt_qualification


def evaluate_blocked_qualification_antigravity(
    qualification: BlockedQualificationAntigravity,
) -> ValidationDecisionAntigravity:
    value = validate_blocked_qualification_semantics_antigravity(qualification)
    passed = all(record.passed for record in value.fold_evidence)
    if passed is not value.passed:
        raise ValueError("qualification pass flag differs from all ten folds")
    decision = ValidationDecisionAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        qualification=value,
        status="stable" if passed else "unstable",
        passed=passed,
    )
    _DECISION_EVIDENCE_BY_ID_ANTIGRAVITY[id(decision)] = (decision, value)
    return decision


def validate_validation_decision_semantics_antigravity(
    value: object,
) -> ValidationDecisionAntigravity:
    if type(value) is not ValidationDecisionAntigravity:
        raise ValueError("validation decision type is invalid")
    evidence = _DECISION_EVIDENCE_BY_ID_ANTIGRAVITY.get(id(value))
    if evidence is None or evidence[0] is not value:
        raise ValueError("validation decision lacks evaluator provenance")
    _stored, qualification = evidence
    expected = evaluate_blocked_qualification_antigravity(qualification)
    if _decision_signature_antigravity(value) != _decision_signature_antigravity(expected):
        raise ValueError("validation decision differs from semantic recomputation")
    return value


def _qualification_for_decision_antigravity(
    decision: ValidationDecisionAntigravity,
) -> BlockedQualificationAntigravity:
    validate_validation_decision_semantics_antigravity(decision)
    return _DECISION_EVIDENCE_BY_ID_ANTIGRAVITY[id(decision)][1]


__all__ = (
    "EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY",
    "_qualification_for_decision_antigravity",
    "evaluate_blocked_qualification_antigravity",
    "validate_blocked_qualification_semantics_antigravity",
    "validate_validation_decision_semantics_antigravity",
)
