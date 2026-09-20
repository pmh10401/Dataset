"""Capture the exact frozen Exp459 fallback authority required by Exp460."""

from __future__ import annotations

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.evaluation_antigravity import (
    EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY,
)
from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
    FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    ConfusionAntigravity,
    Exp459DecisionAntigravity,
    Exp459FallbackAuthorityAntigravity,
    Exp459FoldRecordAntigravity,
    Exp459SeedSummaryAntigravity,
    validate_exp459_fallback_authority_antigravity,
)


def _fold_records_antigravity() -> tuple[Exp459FoldRecordAntigravity, ...]:
    return tuple(
        Exp459FoldRecordAntigravity(
            seed=record.seed,
            fold_index=record.fold_index,
            period_change_auc=record.per_type_auc["period_change"],
            passed=record.passed,
        )
        for record in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence
    )


def _seed_summaries_antigravity(
    fold_records: tuple[Exp459FoldRecordAntigravity, ...],
) -> tuple[Exp459SeedSummaryAntigravity, Exp459SeedSummaryAntigravity]:
    summaries: list[Exp459SeedSummaryAntigravity] = []
    for index, seed in enumerate(FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds):
        start = index * FROZEN_CONFIG_ANTIGRAVITY.folds
        stop = start + FROZEN_CONFIG_ANTIGRAVITY.folds
        records = fold_records[start:stop]
        summaries.append(
            Exp459SeedSummaryAntigravity(
                seed=seed,
                fold_records=records,
                passed=all(record.passed for record in records),
            )
        )
    return tuple(summaries)  # type: ignore[return-value]


def capture_exp459_fallback_authority_antigravity() -> Exp459FallbackAuthorityAntigravity:
    """Return the exact frozen Exp459 scientific result needed by Exp460."""

    fold_records = _fold_records_antigravity()
    authority = Exp459FallbackAuthorityAntigravity(
        fold_records=fold_records,
        seed_summaries=_seed_summaries_antigravity(fold_records),
        decision=Exp459DecisionAntigravity(
            status=FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.decision.status,
            passed=FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.decision.passed,
        ),
        baseline_confusion=ConfusionAntigravity(
            *FROZEN_CONFIG_ANTIGRAVITY.baseline_confusion
        ),
        candidate_confusion=ConfusionAntigravity(
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tp,
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fp,
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.fn,
            EXP458_PREDECESSOR_CONTEXT_ANTIGRAVITY.tn,
        ),
        frozen_authority_digest=FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.canonical_digest,
        official_artifact_map_sha256=FROZEN_CONFIG_ANTIGRAVITY.exp459_official_artifact_map_sha256,
        decision_sha256=FROZEN_CONFIG_ANTIGRAVITY.exp459_decision_sha256,
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count,
    )
    return validate_exp459_fallback_authority_antigravity(authority)


__all__ = ("capture_exp459_fallback_authority_antigravity",)
