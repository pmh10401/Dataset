"""Capture the exact Exp460 predecessor authority required by Exp461."""

from __future__ import annotations

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.frozen_validation_authority_antigravity import (
    FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp460FailedFoldAntigravity,
    Exp460PeriodAuthorityAntigravity,
    _LEAF_SHA256_ANTIGRAVITY,
    _REVIEWED_COMMIT_ANTIGRAVITY,
    validate_exp460_period_authority_antigravity,
)


def _failed_folds_antigravity() -> tuple[Exp460FailedFoldAntigravity, ...]:
    return tuple(
        Exp460FailedFoldAntigravity(
            seed=record.seed,
            fold_index=record.fold_index,
            period_change_auc=record.per_type_auc["period_change"],
        )
        for record in FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence
        if not record.passed
    )


def capture_exp460_period_authority_antigravity() -> Exp460PeriodAuthorityAntigravity:
    """Return the exact frozen Exp460 result snapshot consumed by Exp461."""

    failed_folds = _failed_folds_antigravity()
    authority = Exp460PeriodAuthorityAntigravity(
        predecessor_experiment=FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment,
        truth_commit=FROZEN_CONFIG_ANTIGRAVITY.predecessor_truth_commit,
        run_root=FROZEN_CONFIG_ANTIGRAVITY.predecessor_run_root,
        reviewed_commit=_REVIEWED_COMMIT_ANTIGRAVITY,
        artifact_map_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256,
        dashboard_snapshot_sha256=(
            FROZEN_CONFIG_ANTIGRAVITY.predecessor_dashboard_snapshot_sha256
        ),
        decision_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_decision_sha256,
        status=FROZEN_CONFIG_ANTIGRAVITY.predecessor_status,
        strict_route=FROZEN_CONFIG_ANTIGRAVITY.predecessor_strict_route,
        joint_success=not failed_folds,
        passed_record_count=(
            len(FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY.fold_evidence) - len(failed_folds)
        ),
        failed_fold_count=len(failed_folds),
        worst_period_change_auc=min(row.period_change_auc for row in failed_folds),
        failed_folds=failed_folds,
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        leaf_sha256=_LEAF_SHA256_ANTIGRAVITY,
    )
    return validate_exp460_period_authority_antigravity(authority)


__all__ = ("capture_exp460_period_authority_antigravity",)
