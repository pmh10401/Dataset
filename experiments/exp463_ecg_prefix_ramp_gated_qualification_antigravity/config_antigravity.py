"""Frozen Task 2 inputs for Exp463.

This module contains identity and provenance only.  It deliberately does not
run a scorer, evaluator, or result publisher.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _exact_antigravity(value: object, expected: object) -> bool:
    """Compare frozen values without allowing bool/int or tuple/list aliases."""

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


@dataclass(frozen=True, slots=True)
class HistoricalPeriodThresholdAntigravity:
    seed: int
    fold_index: int
    period_change_auc_min: float

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in (20260721, 20261721):
            raise ValueError("seed must be one of the frozen seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < 5:
            raise ValueError("fold_index must be one of the frozen folds")
        if type(self.period_change_auc_min) is not float or not 0.0 <= self.period_change_auc_min <= 1.0:
            raise ValueError("period_change_auc_min must be a unit float")

    @property
    def period_change_auc(self) -> float:
        return self.period_change_auc_min


_METHOD_DEFINITIONS_ANTIGRAVITY = MappingProxyType(
    {
        "M0": "Exp461 reviewed hydra_period_rank_max score",
        "M1": "front-prefix normalized one-dimensional ramp score",
        "M2": "M1 with 0.995-gated max against M0; sole primary candidate hydra_period_ramp_gated_max",
        "M3": "M1 with always-max combination against M0",
        "M4": "TRAIN-fit per-time median template residual ramp with 0.995-gated max",
        "M5": "whole-series median/MAD normalized ramp with 0.995-gated max",
        "M6": "Exp462 24D LocalGap replay control with always-max combination",
    }
)

_METHOD_SUMMARY_SNAPSHOT_ANTIGRAVITY = (
    # (method, fold_count, passed_count, failed_count, min gradual, min period,
    #  min pooled, max Wilson, finite, passed)
    ("hydra_period_rank_max", 10, 9, 1, 0.8402306222021025, 0.907392736783864, 0.9212266108344651, 0.013439124039399194, True, False),
    ("gradual_signature_local_gap", 10, 0, 10, 0.6966125185730875, 0.8093657260701171, 0.7431184296564559, 0.013439124039399194, True, False),
    ("hydra_period_gradual_rank_max", 10, 3, 7, 0.7953196519289775, 0.8960376741612595, 0.9010220185101353, 0.013439124039399194, True, False),
)

_PREDECESSOR_LEAF_SHA256_ANTIGRAVITY = MappingProxyType(
    {
        "context_antigravity.json": "fd5a5271ae68183d9a121ba46f8dfb18aab8a8399140ee9d224e87af7bfd1291",
        "method_fold_metrics_antigravity.jsonl": "f3d0eab11fa3902c9e4ea30a4d83b19cb5cff52cc3f51a20034c9b3b553ccf0b",
        "failed_folds_antigravity.jsonl": "e7848e96f5380448eb38a21ecfe6e9a19fcca830fead0058b85a285f13da90ea",
        "method_summary_antigravity.json": "0f63f0ee9d1788810fddb27fcb576225c73301098e3ddbf24704710a627eb0a9",
        "decision_antigravity.json": "6eec416030c0b01f945feee824468dc0766dfd036945b1e9e7307e9498d0d58b",
        "dashboard_snapshot_antigravity.json": "d2f1ed5c289aa4eb061c04065cc7b49420f3ca4ebe894b269853703bb36a558b",
        "artifact_sha256_antigravity.json": "5bfdeaeb04013f26987131980f8681afb4eaec2d1aab086cc5ee1220aee69c50",
    }
)

_PRIMARY_MIN_PER_TYPE_AUC_ANTIGRAVITY = MappingProxyType(
    {
        "level_shift": 0.9377483526985564,
        "gradual_drift": 0.7953196519289775,
        "amplitude_scaling": 0.9574971077913585,
        "period_change": 0.8960376741612595,
    }
)


@dataclass(frozen=True, slots=True)
class Exp463ConfigAntigravity:
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
    onset: int
    alpha: float
    gate: float
    change_order: tuple[str, str, str, str]
    method_order: tuple[str, str, str, str, str, str, str]
    method_definitions: Mapping[str, str]
    primary_candidate: str
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    gradual_drift_auc_min: float
    gradual_drift_auc_strictly_above_exp462: float
    period_change_auc_global_min: float
    historical_period_fold_thresholds: tuple[HistoricalPeriodThresholdAntigravity, HistoricalPeriodThresholdAntigravity]
    predecessor_experiment: str
    predecessor_reviewed_code_commit: str
    predecessor_terminal_closeout_commit: str
    predecessor_status: str
    predecessor_primary_candidate: str
    predecessor_passed_record_count: int
    predecessor_required_record_count: int
    predecessor_failed_gradual_drift_auc: float
    predecessor_period_min_auc: float
    predecessor_historical_period_folds: tuple[HistoricalPeriodThresholdAntigravity, HistoricalPeriodThresholdAntigravity]
    predecessor_result_row_count: int
    predecessor_result_failed_row_count: int
    predecessor_result_summary_count: int
    predecessor_result_leaf_count: int
    predecessor_leaf_sha256: Mapping[str, str]
    predecessor_primary_min_per_type_auc: Mapping[str, float]
    predecessor_primary_min_pooled_auc: float
    predecessor_primary_max_normal_wilson_upper: float
    predecessor_result_method_summaries: tuple[tuple[object, ...], tuple[object, ...], tuple[object, ...]]
    label_query_count_must_remain: int
    official_command_invocations_must_remain: int

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp463 configuration")

    @property
    def seed_pair(self) -> tuple[int, int]:
        return self.hydra_seeds

    @property
    def knn_k(self) -> int:
        """Exp461/Exp462 used the reviewed k=3 baseline; retained as provenance only."""
        return 3

    @property
    def method_descriptions(self) -> Mapping[str, str]:
        return self.method_definitions

    @property
    def blocked_fold_count(self) -> int:
        return self.folds

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.normal2_train_rows, self.normal2_width)

    @property
    def confidence_gate(self) -> float:
        return self.gate

    @property
    def predecessor_terminal_commit(self) -> str:
        return self.predecessor_terminal_closeout_commit

    @property
    def predecessor_artifact_map_sha256(self) -> str:
        return self.predecessor_leaf_sha256["artifact_sha256_antigravity.json"]

    @property
    def predecessor_decision_sha256(self) -> str:
        return self.predecessor_leaf_sha256["decision_antigravity.json"]

    @property
    def predecessor_dashboard_snapshot_sha256(self) -> str:
        return self.predecessor_leaf_sha256["dashboard_snapshot_antigravity.json"]

    @property
    def predecessor_summary_sha256(self) -> str:
        return self.predecessor_leaf_sha256["method_summary_antigravity.json"]

    @property
    def method_order_ids(self) -> tuple[str, ...]:
        return self.method_order

    @property
    def result_row_count(self) -> int:
        return self.predecessor_result_row_count

    @property
    def result_summary_count(self) -> int:
        return self.predecessor_result_summary_count

    @property
    def result_leaf_count(self) -> int:
        return self.predecessor_result_leaf_count

    @property
    def result_failed_row_count(self) -> int:
        return self.predecessor_result_failed_row_count

    @property
    def predecessor_result_leaf_sha256(self) -> Mapping[str, str]:
        return self.predecessor_leaf_sha256

    @property
    def predecessor_private_sha256(self) -> Mapping[str, str]:
        return self.predecessor_leaf_sha256


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp463",
        "kind": "ecg_prefix_ramp_gated_qualification",
        "run_identity": "exp463-ecg-prefix-ramp-gated-qualification-antigravity-20260821",
        "official_result_root": "/Users/mac/Documents/Dataset/exp463_ecg_prefix_ramp_gated_qualification_antigravity_20260821",
        "approved_design_sha256": "d5af337a09b11a707198eb0d669a9a6a4b330db44db698daa5ca218cdfbb7efb",
        "dataset": "ECG5000_normal_2",
        "data_split": "TRAIN_only",
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "alpha": 0.005,
        "gate": 0.995,
        "change_order": ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        "method_order": ("M0", "M1", "M2", "M3", "M4", "M5", "M6"),
        "method_definitions": _METHOD_DEFINITIONS_ANTIGRAVITY,
        "primary_candidate": "hydra_period_ramp_gated_max",
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "gradual_drift_auc_min": 0.85,
        "gradual_drift_auc_strictly_above_exp462": 0.8402306222021025,
        "period_change_auc_global_min": 0.907392736783864,
        "historical_period_fold_thresholds": (
            HistoricalPeriodThresholdAntigravity(20260721, 3, 0.907392736783864),
            HistoricalPeriodThresholdAntigravity(20261721, 4, 0.9216148584075248),
        ),
        "predecessor_experiment": "Exp462",
        "predecessor_reviewed_code_commit": "ff57925ae2d5542fe6ec34131686ffa0bc37fe0c",
        "predecessor_terminal_closeout_commit": "040422f62406e3ca19038b9609903c4ba76e1694",
        "predecessor_status": "stopped_before_official_evaluation_qualification_failed",
        "predecessor_primary_candidate": "hydra_period_gradual_rank_max",
        "predecessor_passed_record_count": 3,
        "predecessor_required_record_count": 10,
        "predecessor_failed_gradual_drift_auc": 0.7953196519289775,
        "predecessor_period_min_auc": 0.8960376741612595,
        "predecessor_historical_period_folds": (
            HistoricalPeriodThresholdAntigravity(20260721, 3, 0.8960376741612595),
            HistoricalPeriodThresholdAntigravity(20261721, 4, 0.9285624465570143),
        ),
        "predecessor_result_row_count": 30,
        "predecessor_result_failed_row_count": 18,
        "predecessor_result_summary_count": 3,
        "predecessor_result_leaf_count": 7,
        "predecessor_leaf_sha256": _PREDECESSOR_LEAF_SHA256_ANTIGRAVITY,
        "predecessor_primary_min_per_type_auc": _PRIMARY_MIN_PER_TYPE_AUC_ANTIGRAVITY,
        "predecessor_primary_min_pooled_auc": 0.9010220185101353,
        "predecessor_primary_max_normal_wilson_upper": 0.013439124039399194,
        "predecessor_result_method_summaries": _METHOD_SUMMARY_SNAPSHOT_ANTIGRAVITY,
        "label_query_count_must_remain": 0,
        "official_command_invocations_must_remain": 0,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp463ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = (
    "Exp463ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
    "HistoricalPeriodThresholdAntigravity",
)
