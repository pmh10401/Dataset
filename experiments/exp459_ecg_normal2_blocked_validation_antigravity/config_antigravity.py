"""Frozen result-affecting configuration for Exp459 Task 2."""

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
class Exp459ConfigAntigravity:
    experiment: str
    approved_design_sha256: str
    dataset: str
    kind: str
    run_identity: str
    official_result_root: str
    runtime_executable: str
    runtime_prefix: str
    runtime_python_version: str
    runtime_numpy_version: str
    runtime_aeon_version: str
    predecessor_experiment: str
    predecessor_reviewed_commit: str
    predecessor_artifact_map_sha256: str
    blocked_fold_planner: str
    hydra_seeds: tuple[int, int]
    folds: int
    query_sizes: tuple[int, int, int, int, int]
    query_ranges: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ]
    fit_sizes: tuple[int, int, int, int, int]
    primary_types: tuple[str, str, str, str]
    per_type_auc_min: float
    pooled_auc_min: float
    normal_wilson_upper_max: float
    normal2_train_rows: int
    normal2_width: int
    structure_gate: str
    finite_required: bool
    test_metrics_may_inform_routing: bool
    mlx_applicable: bool

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp459 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp459",
        "approved_design_sha256": "47d530a0cbe7f7cdb0b85f2c0c546b7ede158d8a85824eba8c8caece16b2e275",
        "dataset": "ECG5000_normal_2",
        "kind": "ecg_normal2_blocked_validation",
        "run_identity": "exp459-ecg-normal2-blocked-validation-antigravity-20260818",
        "official_result_root": "/Users/mac/Documents/Dataset/exp459_ecg_normal2_blocked_validation_antigravity_20260818",
        "runtime_executable": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        "runtime_prefix": "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314",
        "runtime_python_version": "3.14.6",
        "runtime_numpy_version": "2.3.5",
        "runtime_aeon_version": "1.4.0",
        "predecessor_experiment": "Exp458",
        "predecessor_reviewed_commit": "fc530cd2b392394aa7ffbd1408bc06d5b1b7c806",
        "predecessor_artifact_map_sha256": "7ef36f232db5d2f2396f3dcdeda15af2e93c34e18a6e05dc411e37fe855737f9",
        "blocked_fold_planner": "numpy.arange+numpy.array_split",
        "hydra_seeds": (20260721, 20261721),
        "folds": 5,
        "query_sizes": (283, 283, 283, 282, 282),
        "query_ranges": ((0, 282), (283, 565), (566, 848), (849, 1130), (1131, 1412)),
        "fit_sizes": (1130, 1130, 1130, 1131, 1131),
        "primary_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "per_type_auc_min": 0.85,
        "pooled_auc_min": 0.90,
        "normal_wilson_upper_max": 0.05,
        "normal2_train_rows": 1413,
        "normal2_width": 140,
        "structure_gate": "added_count > removed_count",
        "finite_required": True,
        "test_metrics_may_inform_routing": False,
        "mlx_applicable": False,
    }
)


FROZEN_CONFIG_ANTIGRAVITY = Exp459ConfigAntigravity(**_FROZEN_VALUES_ANTIGRAVITY)  # type: ignore[arg-type]


__all__ = ("Exp459ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
