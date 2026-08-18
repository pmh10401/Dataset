"""Stable/unstable evaluation for Exp459 blocked validation."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import re

import numpy as np

from .blocked_folds_antigravity import blocked_plan_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BlockedFoldAntigravity,
    BlockedQualificationAntigravity,
    FoldEvidenceAntigravity,
    SeedSummaryAntigravity,
    ValidationDecisionAntigravity,
)


@dataclass(frozen=True, slots=True)
class Exp458PredecessorContextAntigravity:
    experiment: str
    tp: int
    fp: int
    fn: int
    tn: int
    f1: float

    def __post_init__(self) -> None:
        if (
            type(self.experiment) is not str
            or self.experiment != "Exp458"
            or type(self.tp) is not int
            or self.tp != 617
            or type(self.fp) is not int
            or self.fp != 66
            or type(self.fn) is not int
            or self.fn != 829
            or type(self.tn) is not int
            or self.tn != 70788
            or type(self.f1) is not float
        ):
            raise ValueError("Exp458 predecessor metrics differ from the reviewed context")
        expected_f1 = float(Fraction(2 * self.tp, 2 * self.tp + self.fp + self.fn))
        if self.f1 != expected_f1:
            raise ValueError("Exp458 predecessor F1 differs from confusion-matrix recomputation")


EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY = Exp458PredecessorContextAntigravity(
    experiment="Exp458",
    tp=617,
    fp=66,
    fn=829,
    tn=70788,
    f1=0.5796148426491311,
)
_DECISION_EVIDENCE_BY_ID_ANTIGRAVITY: dict[
    int, tuple[ValidationDecisionAntigravity, BlockedQualificationAntigravity]
] = {}


def exp458_predecessor_context_payload_antigravity() -> dict[str, object]:
    value = EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY
    if type(value) is not Exp458PredecessorContextAntigravity:
        raise ValueError("Exp458 predecessor context object type differs")
    rebuilt = Exp458PredecessorContextAntigravity(
        experiment=value.experiment,
        tp=value.tp,
        fp=value.fp,
        fn=value.fn,
        tn=value.tn,
        f1=value.f1,
    )
    return {
        "experiment": rebuilt.experiment,
        "tp": rebuilt.tp,
        "fp": rebuilt.fp,
        "fn": rebuilt.fn,
        "tn": rebuilt.tn,
        "f1": rebuilt.f1,
    }


def _scalar_signature_antigravity(value: object) -> tuple[type, object]:
    return (type(value), value)


def _fold_plan_signature_antigravity(fold: BlockedFoldAntigravity) -> tuple[object, ...]:
    return (
        _scalar_signature_antigravity(fold.n_rows),
        _scalar_signature_antigravity(fold.fold_index),
        fold.fit_indices.tobytes(),
        fold.query_indices.tobytes(),
    )


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


def _json_digest_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _authority_fold_plan_row_antigravity(fold: BlockedFoldAntigravity) -> dict[str, object]:
    return {
        "n_rows": fold.n_rows,
        "fold_index": fold.fold_index,
        "fit_count": len(fold.fit_indices),
        "query_count": len(fold.query_indices),
        "query_start": int(fold.query_indices[0]),
        "query_stop": int(fold.query_indices[-1]),
        "fit_indices": fold.fit_indices.tolist(),
        "query_indices": fold.query_indices.tolist(),
    }


def _authority_fold_evidence_row_antigravity(record: FoldEvidenceAntigravity) -> dict[str, object]:
    return {
        "dataset": record.dataset,
        "seed": record.seed,
        "fold_index": record.fold_index,
        "normal_trials": record.normal_trials,
        "normal_alerts": record.normal_alerts,
        "normal_wilson_upper": record.normal_wilson_upper,
        "per_type_auc": {
            name: record.per_type_auc[name]
            for name in FROZEN_CONFIG_ANTIGRAVITY.primary_types
        },
        "pooled_auc": record.pooled_auc,
        "passed": record.passed,
    }


def _authority_seed_summary_row_antigravity(summary: SeedSummaryAntigravity) -> dict[str, object]:
    return {
        "dataset": summary.dataset,
        "seed": summary.seed,
        "fold_count": len(summary.fold_records),
        "passed": summary.passed,
    }


def _authority_decision_payload_antigravity(
    decision: ValidationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "dataset": decision.dataset,
        "status": decision.status,
        "passed": decision.passed,
    }


def _authority_payload_antigravity(
    fold_plan: tuple[BlockedFoldAntigravity, ...],
    fold_evidence: tuple[FoldEvidenceAntigravity, ...],
    seed_summaries: tuple[SeedSummaryAntigravity, SeedSummaryAntigravity],
    decision: ValidationDecisionAntigravity,
) -> dict[str, object]:
    return {
        "experiment_id": FROZEN_CONFIG_ANTIGRAVITY.experiment,
        "fold_plan": tuple(_authority_fold_plan_row_antigravity(fold) for fold in fold_plan),
        "fold_evidence": tuple(_authority_fold_evidence_row_antigravity(record) for record in fold_evidence),
        "seed_summaries": tuple(_authority_seed_summary_row_antigravity(summary) for summary in seed_summaries),
        "decision": _authority_decision_payload_antigravity(decision),
    }


def _authority_digest_antigravity(
    fold_plan: tuple[BlockedFoldAntigravity, ...],
    fold_evidence: tuple[FoldEvidenceAntigravity, ...],
    seed_summaries: tuple[SeedSummaryAntigravity, SeedSummaryAntigravity],
    decision: ValidationDecisionAntigravity,
) -> str:
    return sha256(
        _json_digest_bytes_antigravity(
            _authority_payload_antigravity(fold_plan, fold_evidence, seed_summaries, decision)
        )
    ).hexdigest()


def _rebuilt_plan_fold_antigravity(fold: object) -> BlockedFoldAntigravity:
    if type(fold) is not BlockedFoldAntigravity:
        raise ValueError("authority fold-plan type is invalid")
    if (
        type(fold.fit_indices) is not np.ndarray
        or fold.fit_indices.dtype != np.dtype(np.int64)
        or fold.fit_indices.flags.writeable
        or type(fold.query_indices) is not np.ndarray
        or fold.query_indices.dtype != np.dtype(np.int64)
        or fold.query_indices.flags.writeable
    ):
        raise ValueError("authority fold-plan arrays must remain exact immutable int64 arrays")
    return BlockedFoldAntigravity(
        n_rows=fold.n_rows,
        fold_index=fold.fold_index,
        fit_indices=fold.fit_indices,
        query_indices=fold.query_indices,
    )


def _validate_authority_fold_plan_antigravity(value: object) -> tuple[BlockedFoldAntigravity, ...]:
    if type(value) is not tuple or len(value) != FROZEN_CONFIG_ANTIGRAVITY.folds:
        raise ValueError("authority fold plan inventory differs")
    rebuilt = tuple(_rebuilt_plan_fold_antigravity(fold) for fold in value)
    expected = blocked_plan_antigravity(FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows)
    if tuple(_fold_plan_signature_antigravity(fold) for fold in rebuilt) != tuple(
        _fold_plan_signature_antigravity(fold) for fold in expected
    ):
        raise ValueError("authority fold plan differs from frozen blocked-plan recomputation")
    return rebuilt


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


@dataclass(frozen=True, slots=True)
class FrozenValidationAuthorityAntigravity:
    fold_plan: tuple[BlockedFoldAntigravity, ...]
    fold_evidence: tuple[FoldEvidenceAntigravity, ...]
    seed_summaries: tuple[SeedSummaryAntigravity, SeedSummaryAntigravity]
    decision: ValidationDecisionAntigravity
    canonical_digest: str

    def __post_init__(self) -> None:
        fold_plan = _validate_authority_fold_plan_antigravity(self.fold_plan)
        qualification = validate_blocked_qualification_semantics_antigravity(
            BlockedQualificationAntigravity(
                dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
                fold_evidence=self.fold_evidence,
                seed_summaries=self.seed_summaries,
                passed=all(summary.passed for summary in self.seed_summaries),
            )
        )
        if type(self.decision) is not ValidationDecisionAntigravity:
            raise ValueError("authority decision type is invalid")
        expected_decision = ValidationDecisionAntigravity(
            dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
            qualification=qualification,
            status="stable" if qualification.passed else "unstable",
            passed=qualification.passed,
        )
        if _decision_signature_antigravity(self.decision) != _decision_signature_antigravity(
            expected_decision
        ):
            raise ValueError("authority decision differs from authority evidence")
        if (
            type(self.canonical_digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", self.canonical_digest) is None
        ):
            raise ValueError("authority canonical digest is invalid")
        expected_digest = _authority_digest_antigravity(
            fold_plan,
            qualification.fold_evidence,
            qualification.seed_summaries,
            expected_decision,
        )
        if self.canonical_digest != expected_digest:
            raise ValueError("authority canonical digest differs")
        object.__setattr__(self, "fold_plan", fold_plan)
        object.__setattr__(self, "fold_evidence", qualification.fold_evidence)
        object.__setattr__(self, "seed_summaries", qualification.seed_summaries)
        object.__setattr__(self, "decision", expected_decision)


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


def freeze_validation_authority_antigravity(
    decision: ValidationDecisionAntigravity,
) -> FrozenValidationAuthorityAntigravity:
    value = validate_validation_decision_semantics_antigravity(decision)
    qualification = _qualification_for_decision_antigravity(value)
    fold_plan = blocked_plan_antigravity(FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows)
    expected_decision = ValidationDecisionAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        qualification=qualification,
        status="stable" if qualification.passed else "unstable",
        passed=qualification.passed,
    )
    digest = _authority_digest_antigravity(
        fold_plan,
        qualification.fold_evidence,
        qualification.seed_summaries,
        expected_decision,
    )
    return FrozenValidationAuthorityAntigravity(
        fold_plan=fold_plan,
        fold_evidence=qualification.fold_evidence,
        seed_summaries=qualification.seed_summaries,
        decision=expected_decision,
        canonical_digest=digest,
    )


def validate_frozen_validation_authority_antigravity(
    authority: object,
) -> FrozenValidationAuthorityAntigravity:
    if type(authority) is not FrozenValidationAuthorityAntigravity:
        raise ValueError("trusted frozen validation authority is required")
    return FrozenValidationAuthorityAntigravity(
        fold_plan=authority.fold_plan,
        fold_evidence=authority.fold_evidence,
        seed_summaries=authority.seed_summaries,
        decision=authority.decision,
        canonical_digest=authority.canonical_digest,
    )


def validate_decision_matches_authority_antigravity(
    decision: ValidationDecisionAntigravity,
    authority: object,
) -> FrozenValidationAuthorityAntigravity:
    trusted = validate_frozen_validation_authority_antigravity(authority)
    value = validate_validation_decision_semantics_antigravity(decision)
    if _decision_signature_antigravity(value) != _decision_signature_antigravity(trusted.decision):
        raise ValueError("validation decision differs from trusted authority")
    return trusted


__all__ = (
    "EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY",
    "Exp458PredecessorContextAntigravity",
    "FrozenValidationAuthorityAntigravity",
    "_qualification_for_decision_antigravity",
    "evaluate_blocked_qualification_antigravity",
    "exp458_predecessor_context_payload_antigravity",
    "freeze_validation_authority_antigravity",
    "validate_blocked_qualification_semantics_antigravity",
    "validate_decision_matches_authority_antigravity",
    "validate_frozen_validation_authority_antigravity",
    "validate_validation_decision_semantics_antigravity",
)
