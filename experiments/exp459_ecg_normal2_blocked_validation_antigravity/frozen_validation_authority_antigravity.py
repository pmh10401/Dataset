"""Frozen Exp459 Task 5 blocked-validation authority from exact private equality."""

from __future__ import annotations

from .blocked_folds_antigravity import blocked_plan_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    BlockedQualificationAntigravity,
    FoldEvidenceAntigravity,
    SeedSummaryAntigravity,
    ValidationDecisionAntigravity,
)
from .evaluation_antigravity import FrozenValidationAuthorityAntigravity


_FOLD_PLAN_ANTIGRAVITY = blocked_plan_antigravity(
    FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows
)


def _fold_evidence_antigravity(
    *,
    seed: int,
    fold_index: int,
    normal_trials: int,
    normal_alerts: int,
    normal_wilson_upper: float,
    level_shift_auc: float,
    gradual_drift_auc: float,
    amplitude_scaling_auc: float,
    period_change_auc: float,
    pooled_auc: float,
    passed: bool,
) -> FoldEvidenceAntigravity:
    fold = _FOLD_PLAN_ANTIGRAVITY[fold_index]
    return FoldEvidenceAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed=seed,
        fold_index=fold_index,
        fit_indices=fold.fit_indices,
        query_indices=fold.query_indices,
        normal_trials=normal_trials,
        normal_alerts=normal_alerts,
        normal_wilson_upper=normal_wilson_upper,
        per_type_auc={
            "level_shift": level_shift_auc,
            "gradual_drift": gradual_drift_auc,
            "amplitude_scaling": amplitude_scaling_auc,
            "period_change": period_change_auc,
        },
        pooled_auc=pooled_auc,
        passed=passed,
    )


_FOLD_EVIDENCE_ANTIGRAVITY = (
    _fold_evidence_antigravity(
        seed=20260721,
        fold_index=0,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.9740663511843075,
        gradual_drift_auc=0.8995492514577532,
        amplitude_scaling_auc=0.9918840290177178,
        period_change_auc=0.878922199053553,
        pooled_auc=0.9361054576783329,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20260721,
        fold_index=1,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.9838804330182672,
        gradual_drift_auc=0.9128219855410855,
        amplitude_scaling_auc=0.9993756945398244,
        period_change_auc=0.9676734632721098,
        pooled_auc=0.9659378940928217,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20260721,
        fold_index=2,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.9801595724756209,
        gradual_drift_auc=0.8976014184220055,
        amplitude_scaling_auc=0.9992383473385859,
        period_change_auc=0.9438249946934036,
        pooled_auc=0.955206083232404,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20260721,
        fold_index=3,
        normal_trials=282,
        normal_alerts=0,
        normal_wilson_upper=0.013439124039399194,
        level_shift_auc=0.9671419948694734,
        gradual_drift_auc=0.8858960816860318,
        amplitude_scaling_auc=0.990304813641165,
        period_change_auc=0.8208339620743423,
        pooled_auc=0.9160442130677531,
        passed=False,
    ),
    _fold_evidence_antigravity(
        seed=20260721,
        fold_index=4,
        normal_trials=282,
        normal_alerts=0,
        normal_wilson_upper=0.013439124039399194,
        level_shift_auc=0.9888335596800966,
        gradual_drift_auc=0.9243247321563302,
        amplitude_scaling_auc=0.996290428046879,
        period_change_auc=0.8865625471555757,
        pooled_auc=0.9490028167597203,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20261721,
        fold_index=0,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.974615739989262,
        gradual_drift_auc=0.8880245726629125,
        amplitude_scaling_auc=0.9964040005493888,
        period_change_auc=0.8684588395410107,
        pooled_auc=0.9318757881856434,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20261721,
        fold_index=1,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.9793729475957996,
        gradual_drift_auc=0.9066913059221616,
        amplitude_scaling_auc=0.9973030004120416,
        period_change_auc=0.9500056187491416,
        pooled_auc=0.9583432181697861,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20261721,
        fold_index=2,
        normal_trials=283,
        normal_alerts=0,
        normal_wilson_upper=0.01339227194174688,
        level_shift_auc=0.9755272259611183,
        gradual_drift_auc=0.8859144202075191,
        amplitude_scaling_auc=0.9998501666895578,
        period_change_auc=0.8542871055950255,
        pooled_auc=0.9288947296133052,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20261721,
        fold_index=3,
        normal_trials=282,
        normal_alerts=0,
        normal_wilson_upper=0.013439124039399194,
        level_shift_auc=0.9634827221970725,
        gradual_drift_auc=0.8799733413812183,
        amplitude_scaling_auc=0.9869347618329057,
        period_change_auc=0.9407725969518635,
        pooled_auc=0.942790855590765,
        passed=True,
    ),
    _fold_evidence_antigravity(
        seed=20261721,
        fold_index=4,
        normal_trials=282,
        normal_alerts=0,
        normal_wilson_upper=0.013439124039399194,
        level_shift_auc=0.9837533323273477,
        gradual_drift_auc=0.9191439062421407,
        amplitude_scaling_auc=0.993297620844022,
        period_change_auc=0.8004124541019063,
        pooled_auc=0.9241518283788541,
        passed=False,
    ),
)

_SEED_SUMMARIES_ANTIGRAVITY = (
    SeedSummaryAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed=20260721,
        fold_records=_FOLD_EVIDENCE_ANTIGRAVITY[0:5],
        passed=False,
    ),
    SeedSummaryAntigravity(
        dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
        seed=20261721,
        fold_records=_FOLD_EVIDENCE_ANTIGRAVITY[5:10],
        passed=False,
    ),
)

_QUALIFICATION_ANTIGRAVITY = BlockedQualificationAntigravity(
    dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
    fold_evidence=_FOLD_EVIDENCE_ANTIGRAVITY,
    seed_summaries=_SEED_SUMMARIES_ANTIGRAVITY,
    passed=False,
)

_DECISION_ANTIGRAVITY = ValidationDecisionAntigravity(
    dataset=FROZEN_CONFIG_ANTIGRAVITY.dataset,
    qualification=_QUALIFICATION_ANTIGRAVITY,
    status="unstable",
    passed=False,
)

FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY = FrozenValidationAuthorityAntigravity(
    fold_plan=_FOLD_PLAN_ANTIGRAVITY,
    fold_evidence=_FOLD_EVIDENCE_ANTIGRAVITY,
    seed_summaries=_SEED_SUMMARIES_ANTIGRAVITY,
    decision=_DECISION_ANTIGRAVITY,
    canonical_digest="f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
)


__all__ = ("FROZEN_VALIDATION_AUTHORITY_ANTIGRAVITY",)
