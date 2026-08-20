"""Capture the exact Exp461 predecessor authority required by Exp462."""

from __future__ import annotations

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp461GradualAuthorityAntigravity,
    Exp461HistoricalPeriodFoldAntigravity,
    _PRIVATE_SHA256_ANTIGRAVITY,
    validate_exp461_gradual_authority_antigravity,
)


def capture_exp461_gradual_authority_antigravity() -> Exp461GradualAuthorityAntigravity:
    """Return the exact frozen Exp461 terminal snapshot consumed by Exp462."""

    authority = Exp461GradualAuthorityAntigravity(
        predecessor_experiment=FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment,
        terminal_commit=FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_commit,
        artifact_map_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256,
        decision_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_decision_sha256,
        dashboard_snapshot_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_dashboard_snapshot_sha256,
        summary_sha256=FROZEN_CONFIG_ANTIGRAVITY.predecessor_summary_sha256,
        status=FROZEN_CONFIG_ANTIGRAVITY.predecessor_status,
        primary_candidate=FROZEN_CONFIG_ANTIGRAVITY.predecessor_primary_candidate,
        passed_record_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_passed_record_count,
        required_record_count=FROZEN_CONFIG_ANTIGRAVITY.predecessor_required_record_count,
        failed_gradual_drift_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_failed_gradual_drift_auc,
        period_min_auc=FROZEN_CONFIG_ANTIGRAVITY.predecessor_period_min_auc,
        historical_period_folds=(
            Exp461HistoricalPeriodFoldAntigravity(
                seed=20260721,
                fold_index=3,
                period_change_auc=0.907392736783864,
            ),
            Exp461HistoricalPeriodFoldAntigravity(
                seed=20261721,
                fold_index=4,
                period_change_auc=0.9216148584075248,
            ),
        ),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        private_sha256=_PRIVATE_SHA256_ANTIGRAVITY,
    )
    return validate_exp461_gradual_authority_antigravity(authority)


__all__ = ("capture_exp461_gradual_authority_antigravity",)
