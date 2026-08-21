"""Frozen Task 2 inputs for Exp464."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    if isinstance(expected, MappingProxyType):
        if tuple(value) != tuple(expected):  # type: ignore[arg-type]
            return False
        return all(  # type: ignore[arg-type]
            _exact_antigravity(value[key], expected[key])  # type: ignore[index]
            for key in expected
        )
    return value == expected


def _validate_summary_float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class HistoricalPeriodThresholdAntigravity:
    seed: int
    fold_index: int
    period_change_auc_min: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (20260721, 20261721):
            raise ValueError("seed must be one of the frozen seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < 5:
            raise ValueError("fold_index must be one of the frozen blocked folds")
        _validate_summary_float_antigravity(
            self.period_change_auc_min,
            "period_change_auc_min",
        )


@dataclass(frozen=True, slots=True)
class Exp463MethodSummarySnapshotAntigravity:
    method: str
    fold_count: int
    passed_count: int
    failed_count: int
    min_gradual_drift_auc: float
    min_period_change_auc: float
    min_pooled_auc: float
    max_normal_wilson_upper: float
    all_scores_finite: bool

    def __post_init__(self) -> None:
        if type(self.method) is not str or self.method not in ("C0", "C1", "C2", "C7"):
            raise ValueError("method must be one of the frozen Exp463 control IDs")
        if type(self.fold_count) is not int or self.fold_count <= 0:
            raise ValueError("fold_count must be a positive integer")
        if type(self.passed_count) is not int or self.passed_count < 0:
            raise ValueError("passed_count must be a non-negative integer")
        if type(self.failed_count) is not int or self.failed_count < 0:
            raise ValueError("failed_count must be a non-negative integer")
        if self.fold_count != self.passed_count + self.failed_count:
            raise ValueError("fold_count must equal passed_count + failed_count")
        _validate_summary_float_antigravity(
            self.min_gradual_drift_auc,
            "min_gradual_drift_auc",
        )
        _validate_summary_float_antigravity(
            self.min_period_change_auc,
            "min_period_change_auc",
        )
        _validate_summary_float_antigravity(self.min_pooled_auc, "min_pooled_auc")
        _validate_summary_float_antigravity(
            self.max_normal_wilson_upper,
            "max_normal_wilson_upper",
        )
        if type(self.all_scores_finite) is not bool:
            raise ValueError("all_scores_finite must be bool")


_METHOD_DEFINITIONS_ANTIGRAVITY = MappingProxyType(
    {
        "C0": "Exp463 M0 exact replay control",
        "C1": "Exp463 M2 exact replay control",
        "C2": "Exp463 M4 exact replay control",
        "C3": "template residual centered ramp rank only without M0 max or cosine gate",
        "C4": "same C3 rank with unconditional max against C0 and no cosine gate",
        "C5": "centered ramp confirmation without template subtraction",
        "C6": "template residual centered ramp rank with cosine >= 0.95 confirmed max against C0",
        "C7": "Exp463 M6 exact replay control",
        "C8": "template residual ramp confirmation without centering",
    }
)

_CONTROL_SUMMARIES_ANTIGRAVITY = MappingProxyType(
    {
        "C0": Exp463MethodSummarySnapshotAntigravity(
            method="C0",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8402306222021025,
            min_period_change_auc=0.907392736783864,
            min_pooled_auc=0.9212266108344651,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C1": Exp463MethodSummarySnapshotAntigravity(
            method="C1",
            fold_count=10,
            passed_count=7,
            failed_count=3,
            min_gradual_drift_auc=0.83450279161008,
            min_period_change_auc=0.8984897640963734,
            min_pooled_auc=0.9128093405764297,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C2": Exp463MethodSummarySnapshotAntigravity(
            method="C2",
            fold_count=10,
            passed_count=9,
            failed_count=1,
            min_gradual_drift_auc=0.8399225391076908,
            min_period_change_auc=0.9044753785020874,
            min_pooled_auc=0.9193733967104271,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
        "C7": Exp463MethodSummarySnapshotAntigravity(
            method="C7",
            fold_count=10,
            passed_count=3,
            failed_count=7,
            min_gradual_drift_auc=0.7953196519289775,
            min_period_change_auc=0.8960376741612595,
            min_pooled_auc=0.9010220185101353,
            max_normal_wilson_upper=0.013439124039399194,
            all_scores_finite=True,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Exp464ConfigAntigravity:
    experiment: str
    kind: str
    run_identity: str
    official_result_root: str
    approved_design_sha256: str
    fixed_interpreter: str
    dataset: str
    data_split: str
    normal2_train_rows: int
    normal2_width: int
    hydra_seeds: tuple[int, int]
    folds: int
    onset: int
    cosine_min: float
    method_order: tuple[str, str, str, str, str, str, str, str, str]
    method_definitions: Mapping[str, str]
    primary_candidate: str
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    gradual_drift_auc_min: float
    gradual_drift_auc_strictly_above_exp463: float
    period_change_auc_global_min: float
    historical_period_fold_thresholds: tuple[
        HistoricalPeriodThresholdAntigravity,
        HistoricalPeriodThresholdAntigravity,
    ]
    predecessor_experiment: str
    predecessor_terminal_closeout_commit: str
    predecessor_task5_report_sha256: str
    predecessor_private_run_count: int
    predecessor_result_row_count: int
    predecessor_result_failed_row_count: int
    predecessor_result_summary_count: int
    predecessor_result_leaf_count: int
    predecessor_artifact_map_sha256: str
    predecessor_control_summaries: Mapping[str, Exp463MethodSummarySnapshotAntigravity]
    label_query_count_must_remain: int
    official_evaluate_count_must_remain: int
    official_inspect_count_must_remain: int

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp464 configuration")

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.normal2_train_rows, self.normal2_width)

    @property
    def blocked_fold_count(self) -> int:
        return self.folds

    @property
    def seed_pair(self) -> tuple[int, int]:
        return self.hydra_seeds


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp464",
        "kind": "ecg_residual_centered_ramp_confirmation_qualification",
        "run_identity": "exp464-ecg-residual-centered-ramp-confirmation-antigravity-20260821",
        "official_result_root": "/Users/mac/Documents/Dataset/exp464_ecg_residual_centered_ramp_confirmation_antigravity_20260821",
        "approved_design_sha256": "d0e36ae176598810289fbd5693a5e582ca6f0821a8b8fcd52edd1bcb96ec1f9f",
        "fixed_interpreter": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        "dataset": "ECG5000_normal_2",
        "data_split": "TRAIN_only",
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "cosine_min": 0.95,
        "method_order": ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"),
        "method_definitions": _METHOD_DEFINITIONS_ANTIGRAVITY,
        "primary_candidate": "template_residual_centered_ramp_confirmed_max",
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "gradual_drift_auc_min": 0.85,
        "gradual_drift_auc_strictly_above_exp463": 0.8402306222021025,
        "period_change_auc_global_min": 0.907392736783864,
        "historical_period_fold_thresholds": (
            HistoricalPeriodThresholdAntigravity(
                seed=20260721,
                fold_index=3,
                period_change_auc_min=0.907392736783864,
            ),
            HistoricalPeriodThresholdAntigravity(
                seed=20261721,
                fold_index=4,
                period_change_auc_min=0.9216148584075248,
            ),
        ),
        "predecessor_experiment": "Exp463",
        "predecessor_terminal_closeout_commit": "147225831e9c13e9df5043fa88757bb03e01b79a",
        "predecessor_task5_report_sha256": "e43abf66e43921941dece398514858be91e39d121c1173b26b352b67ba78e8e5",
        "predecessor_private_run_count": 2,
        "predecessor_result_row_count": 70,
        "predecessor_result_failed_row_count": 30,
        "predecessor_result_summary_count": 7,
        "predecessor_result_leaf_count": 7,
        "predecessor_artifact_map_sha256": "25fb4ab237ae8f2601c2bc03b05f4e6a2d0cb9889876a83fa0f2c4ee7c4f0ea5",
        "predecessor_control_summaries": _CONTROL_SUMMARIES_ANTIGRAVITY,
        "label_query_count_must_remain": 0,
        "official_evaluate_count_must_remain": 0,
        "official_inspect_count_must_remain": 0,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp464ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = (
    "Exp463MethodSummarySnapshotAntigravity",
    "Exp464ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
    "HistoricalPeriodThresholdAntigravity",
)
