"""Frozen result-affecting configuration for Exp458 Task 2."""

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
class Exp458ConfigAntigravity:
    experiment: str
    approved_design_sha256: str
    dataset: str
    kind: str
    run_identity: str
    official_result_root: str
    hydra_seeds: tuple[int, int]
    folds: int
    primary_types: tuple[str, str, str, str]
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    structure_gate: str
    finite_required: bool
    test_metrics_may_inform_routing: bool
    mlx_applicable: bool

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp458 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp458",
        "approved_design_sha256": "52abe997c0e14badc2c886b888e3b2b2074cf002216891f173ab3b1dd6d7ea5d",
        "dataset": "ECG5000_normal_2",
        "kind": "ecg_normal2_rank_confidence_routing",
        "run_identity": "exp458-ecg-normal2-rank-confidence-routing-antigravity-20260818",
        "official_result_root": "/Users/mac/Documents/Dataset/exp458_ecg_normal2_rank_confidence_routing_antigravity_20260818",
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "primary_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "structure_gate": "added_count > removed_count",
        "finite_required": True,
        "test_metrics_may_inform_routing": False,
        "mlx_applicable": False,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp458ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp458ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
