"""Frozen result-affecting configuration for Exp457."""

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
class Exp457ConfigAntigravity:
    experiment: str
    approved_design_sha256: str
    ecg_datasets: tuple[str, ...]
    candidate: str
    chlorine_route: str
    hydra_seeds: tuple[int, ...]
    folds: int
    alpha: float
    knn_k: int
    kernels: int
    groups: int
    primary_types: tuple[str, ...]
    normal_wilson_upper_max: float
    pooled_sensitivity_min: float
    per_type_sensitivity_min: float
    finite_required: bool
    structure_gate: str
    final_fn_max: int
    final_fp_max: int
    final_f1_strictly_above: float
    test_metrics_may_inform_routing: bool
    mlx_applicable: bool

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp457 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType({
    "experiment": "Exp457",
    "approved_design_sha256": "3b702e0224dd86e1d0c93268b8a2e2691ea3858cbd30a4792f9c6539c0424e20",
    "ecg_datasets": tuple(f"ECG5000_normal_{index}" for index in range(1, 6)),
    "candidate": "hydra_local_gap",
    "chlorine_route": "multirocket_local_gap",
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
    "normal_wilson_upper_max": 0.05,
    "pooled_sensitivity_min": 0.70,
    "per_type_sensitivity_min": 0.50,
    "finite_required": True,
    "structure_gate": "added_count > removed_count",
    "final_fn_max": 834,
    "final_fp_max": 66,
    "final_f1_strictly_above": 0.5764150943396227,
    "test_metrics_may_inform_routing": False,
    "mlx_applicable": False,
})


FROZEN_CONFIG_ANTIGRAVITY = Exp457ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp457ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
