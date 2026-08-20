"""Frozen Task 2 inputs for Exp461."""

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
class Exp461ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    official_result_root: str
    runtime_executable: str
    approved_design_sha256: str
    preregistration_sha256: str
    runbook_sha256: str
    dataset: str
    data_split: str
    normal2_train_rows: int
    normal2_width: int
    hydra_seeds: tuple[int, int]
    folds: int
    alpha: float
    knn_k: int
    kernels: int
    groups: int
    primary_types: tuple[str, str, str, str]
    method_order: tuple[str, str, str]
    primary_candidate: str
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    predecessor_experiment: str
    predecessor_truth_commit: str
    predecessor_run_root: str
    predecessor_artifact_map_sha256: str
    predecessor_dashboard_snapshot_sha256: str
    predecessor_decision_sha256: str
    predecessor_status: str
    predecessor_strict_route: str
    predecessor_passed_record_count: int
    predecessor_failed_fold_count: int
    predecessor_worst_period_change_auc: float
    label_query_count_must_remain: int
    official_command_invocations_must_remain: int

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp461 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp461",
        "kind": "ecg_period_signature_qualification",
        "run_identity": "exp461-ecg-period-signature-qualification-antigravity-20260820",
        "official_result_root": "/Users/mac/Documents/Dataset/exp461_ecg_period_signature_qualification_antigravity_20260820",
        "runtime_executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        "approved_design_sha256": "3b1d6bc519a4acc000f45f30ba8ae5d610dcd0d2aa5e1cb61bd53837ba28f2c2",
        "preregistration_sha256": "9595d1f898b309ac899c821867a2f38e6f0eb24f147e6888970e1095534490cb",
        "runbook_sha256": "3c9ce4c884ae0f5d7c9551b508a8e3092899f481fc3bc1a14adbd92f81f87807",
        "dataset": "ECG5000_normal_2",
        "data_split": "TRAIN_only",
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "alpha": 0.005,
        "knn_k": 3,
        "kernels": 8,
        "groups": 64,
        "primary_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "method_order": (
            "hydra_local_gap",
            "period_signature_local_gap",
            "hydra_period_rank_max",
        ),
        "primary_candidate": "hydra_period_rank_max",
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "predecessor_experiment": "Exp460",
        "predecessor_truth_commit": "67d39f5c0a75e061e2d107fc9a9f58f52e76c2f1",
        "predecessor_run_root": "/Users/mac/Documents/Dataset/exp460_ecg_strict_fallback_antigravity_20260820",
        "predecessor_artifact_map_sha256": "8286c4da86d3f180e8b5e20cf54fc6f0011a26cf54d92b1a4867f68e5f44d916",
        "predecessor_dashboard_snapshot_sha256": "fea8b4e43d256feddef48545e201f674bfe39fe85559050ed405f4e996709ee5",
        "predecessor_decision_sha256": "0859be73fb8b7c77cc803d67745ecd1bd42766ea5e3c005ce1bb9d2d3e6342c1",
        "predecessor_status": "fallback_triggered_performance_not_improved",
        "predecessor_strict_route": "baseline",
        "predecessor_passed_record_count": 8,
        "predecessor_failed_fold_count": 2,
        "predecessor_worst_period_change_auc": 0.8004124541019063,
        "label_query_count_must_remain": 0,
        "official_command_invocations_must_remain": 0,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp461ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp461ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
