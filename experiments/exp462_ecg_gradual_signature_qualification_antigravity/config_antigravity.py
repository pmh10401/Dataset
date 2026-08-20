"""Frozen Task 2 inputs for Exp462."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is tuple:
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


@dataclass(frozen=True, slots=True)
class Exp462ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    official_result_root: str
    approved_design_sha256: str
    dataset: str
    data_split: str
    normal2_train_rows: int
    normal2_width: int
    hydra_seeds: tuple[int, int]
    folds: int
    alpha: float
    knn_k: int
    change_order: tuple[str, str, str, str]
    method_order: tuple[str, str, str]
    primary_candidate: str
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    gradual_drift_auc_min: float
    period_change_auc_global_min: float
    predecessor_experiment: str
    predecessor_reviewed_code_commit: str
    predecessor_terminal_closeout_commit: str
    predecessor_artifact_map_sha256: str
    predecessor_decision_sha256: str
    predecessor_dashboard_snapshot_sha256: str
    predecessor_summary_sha256: str
    predecessor_status: str
    predecessor_primary_candidate: str
    predecessor_passed_record_count: int
    predecessor_required_record_count: int
    predecessor_failed_gradual_drift_auc: float
    predecessor_period_min_auc: float
    label_query_count_must_remain: int
    official_command_invocations_must_remain: int

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp462 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp462",
        "kind": "ecg_gradual_signature_qualification",
        "run_identity": "exp462-ecg-gradual-signature-qualification-antigravity-20260820",
        "official_result_root": "/Users/mac/Documents/Dataset/exp462_ecg_gradual_signature_qualification_antigravity_20260820",
        "approved_design_sha256": "34a950e619b7ddd1accc7cc41281a476611010f27ef02b1a73fa46523d064829",
        "dataset": "ECG5000_normal_2",
        "data_split": "TRAIN_only",
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "alpha": 0.005,
        "knn_k": 3,
        "change_order": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "method_order": (
            "hydra_period_rank_max",
            "gradual_signature_local_gap",
            "hydra_period_gradual_rank_max",
        ),
        "primary_candidate": "hydra_period_gradual_rank_max",
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "gradual_drift_auc_min": 0.85,
        "period_change_auc_global_min": 0.907392736783864,
        "predecessor_experiment": "Exp461",
        "predecessor_reviewed_code_commit": "3a306ff3e276dae13b28b9733db30e9934da7eb2",
        "predecessor_terminal_closeout_commit": "0340b6ee737da84777283668449738d0c1585566",
        "predecessor_artifact_map_sha256": "8d0992d362cefab305ef8801afd9d1f5d51522e623fa372ba18f6d5d0f3478be",
        "predecessor_decision_sha256": "27f6d8db82b71051a16b68d3ad2f69f6608f6a922d765b35029e594afc581315",
        "predecessor_dashboard_snapshot_sha256": "e0e70dd42536f71aa736a8c178d14ed84b268f703623937dfe1f8663967bbabc",
        "predecessor_summary_sha256": "7500d2599060b257a171856fb68bed39a0ec0f4ddb26ca6084e606c911251e25",
        "predecessor_status": "qualification_failed_before_official_evaluation",
        "predecessor_primary_candidate": "hydra_period_rank_max",
        "predecessor_passed_record_count": 9,
        "predecessor_required_record_count": 10,
        "predecessor_failed_gradual_drift_auc": 0.8402306222021025,
        "predecessor_period_min_auc": 0.907392736783864,
        "label_query_count_must_remain": 0,
        "official_command_invocations_must_remain": 0,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp462ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp462ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
