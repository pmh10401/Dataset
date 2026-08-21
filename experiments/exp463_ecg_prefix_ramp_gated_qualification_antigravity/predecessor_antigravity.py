"""Capture the exact Exp462 terminal snapshot consumed by Exp463."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp462AuthorityAntigravity,
    Exp462MethodSummaryAntigravity,
    Exp462ResultRowIdentityAntigravity,
    Exp462ResultSnapshotAntigravity,
    Exp463HistoricalPeriodFoldAntigravity,
    _EXP462_METHODS_ANTIGRAVITY,
    validate_exp462_authority_antigravity,
)


def _result_snapshot_antigravity() -> Exp462ResultSnapshotAntigravity:
    rows = tuple(
        Exp462ResultRowIdentityAntigravity(method, seed, fold_index)
        for method in _EXP462_METHODS_ANTIGRAVITY
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    summaries = tuple(
        Exp462MethodSummaryAntigravity(
            method=str(snapshot[0]),
            fold_count=int(snapshot[1]),
            passed_count=int(snapshot[2]),
            failed_count=int(snapshot[3]),
            min_gradual_drift_auc=float(snapshot[4]),
            min_period_change_auc=float(snapshot[5]),
            min_pooled_auc=float(snapshot[6]),
            max_normal_wilson_upper=float(snapshot[7]),
            all_scores_finite=bool(snapshot[8]),
            passed=bool(snapshot[9]),
        )
        for snapshot in FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_method_summaries
    )
    return Exp462ResultSnapshotAntigravity(
        row_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_row_count,
        failed_row_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_failed_row_count,
        summary_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_summary_count,
        leaf_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_result_leaf_count,
        rows=rows,
        summaries=summaries,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate,
        primary_passed_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count,
        primary_required_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count,
        primary_min_per_type_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_min_per_type_auc,
        primary_min_pooled_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_min_pooled_auc,
        primary_max_normal_wilson_upper=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_max_normal_wilson_upper,
        leaf_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256,
    )


def capture_exp462_authority_antigravity() -> Exp462AuthorityAntigravity:
    """Return source-free, immutable Exp462 terminal provenance and results."""

    authority = Exp462AuthorityAntigravity(
        predecessor_experiment=FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment,
        reviewed_code_commit=FROZEN_CONFIG_ANTIGRAVITY.predecessor_reviewed_code_commit,
        terminal_closeout_commit=FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_closeout_commit,
        status=FROZEN_CONFIG_ANTIGRAVITY.predecessor_status,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate,
        passed_record_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count,
        required_record_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count,
        failed_gradual_drift_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc,
        period_min_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_period_min_auc,
        historical_period_folds=tuple(
            Exp463HistoricalPeriodFoldAntigravity(
                row.seed,
                row.fold_index,
                row.period_change_auc_min,
            )
            for row in FROZEN_CONFIG_ANTIGRAVITY.predecessor_historical_period_folds
        ),
        result_snapshot=_result_snapshot_antigravity(),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        private_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_leaf_sha256,
    )
    return validate_exp462_authority_antigravity(authority)


# Future phases may use the descriptive alias; both names resolve to the same
# strict source-free capture and do not add another data path.
capture_exp462_gradual_authority_antigravity = capture_exp462_authority_antigravity


__all__ = (
    "capture_exp462_authority_antigravity",
    "capture_exp462_gradual_authority_antigravity",
)
