"""Frozen Exp468 ECG200 local-STFT scientific authority.

``hann_definition`` names the approved symmetric-Hann reference
``w[n] = 0.5 - 0.5*cos(2*pi*n/27)``. It does not select a runtime backend;
Numba/MLX assessment belongs to Task 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(  # type: ignore[arg-type]
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)  # type: ignore[arg-type]
        )
    return value == expected


@dataclass(frozen=True, slots=True)
class Exp468ConfigAntigravity:
    experiment: str
    dataset: str
    data_split: str
    train_rows: int
    width: int
    normal_rows: int
    train_md5: str
    interpolated_width: int
    seeds: tuple[int, int]
    folds: int
    onset: int
    suffix_length: int
    window_starts: tuple[int, int, int, int, int]
    window_length: int
    hop_length: int
    rfft_length: int
    non_dc_bins: int
    power_epsilon: float
    hann_definition: str
    js_log_base: str
    change_types: tuple[str, str, str, str]
    method_order: tuple[str, str, str]
    primary_candidate: str
    period_auc_min: float
    required_fold_results: int
    normal_trials_per_seed: int
    approved_design_sha256: str

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(
                    f"{name} differs from the frozen Exp468 configuration",
                )

    @property
    def train_shape(self) -> tuple[int, int]:
        return (self.train_rows, self.width)


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "experiment": "Exp468",
        "dataset": "ECG200",
        "data_split": "TRAIN_only",
        "train_rows": 100,
        "width": 96,
        "normal_rows": 69,
        "train_md5": "808875aeac1c358a90c630fe369327c4",
        "interpolated_width": 140,
        "seeds": (20260721, 20261721),
        "folds": 5,
        "onset": 56,
        "suffix_length": 84,
        "window_starts": (0, 14, 28, 42, 56),
        "window_length": 28,
        "hop_length": 14,
        "rfft_length": 28,
        "non_dc_bins": 14,
        "power_epsilon": 1e-12,
        "hann_definition": "numpy_hanning_symmetric",
        "js_log_base": "e",
        "change_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "method_order": ("C0", "S10", "C10"),
        "primary_candidate": "C10",
        "period_auc_min": 0.85,
        "required_fold_results": 10,
        "normal_trials_per_seed": 69,
        "approved_design_sha256": (
            "fa799dd628028163b6bc2dee5af6fd3f507b555849a00b194e32ecdc3f7078b7"
        ),
    }
)

FROZEN_CONFIG_ANTIGRAVITY = Exp468ConfigAntigravity(
    **dict(_FROZEN_VALUES_ANTIGRAVITY),
)

__all__ = (
    "Exp468ConfigAntigravity",
    "FROZEN_CONFIG_ANTIGRAVITY",
)
