"""Frozen Exp471 ACF peak-lag-shift authority; scoring is deliberately out of scope."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType


def _exact_antigravity(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, tuple):
        return len(value) == len(expected) and all(
            _exact_antigravity(item, wanted)
            for item, wanted in zip(value, expected, strict=True)
        )
    return value == expected


@dataclass(frozen=True, slots=True)
class Exp471ConfigAntigravity:
    seeds: tuple[int, int] = (20260721, 20261721)
    folds: int = 5
    normal_rows: int = 69
    source_width: int = 96
    interpolated_width: int = 140
    onset: int = 56
    warp_factor: float = 1.25
    early_start: int = 0
    late_start: int = 84
    window_length: int = 56
    acf_lag_min: int = 1
    acf_lag_max: int = 25
    peak_lag_min: int = 2
    peak_lag_max: int = 24
    mad_scale: float = 1.4826
    method_order: tuple[str, str, str] = ("C0", "E12", "C12")
    change_types: tuple[str, str, str, str] = (
        "level_shift",
        "gradual_drift",
        "amplitude_scaling",
        "period_change",
    )
    required_speedup: float = 1.1

    def __post_init__(self) -> None:
        for name, expected in _FROZEN_VALUES_ANTIGRAVITY.items():
            if not _exact_antigravity(getattr(self, name), expected):
                raise ValueError(f"{name} differs from the frozen Exp471 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "seeds": (20260721, 20261721),
        "folds": 5,
        "normal_rows": 69,
        "source_width": 96,
        "interpolated_width": 140,
        "onset": 56,
        "warp_factor": 1.25,
        "early_start": 0,
        "late_start": 84,
        "window_length": 56,
        "acf_lag_min": 1,
        "acf_lag_max": 25,
        "peak_lag_min": 2,
        "peak_lag_max": 24,
        "mad_scale": 1.4826,
        "method_order": ("C0", "E12", "C12"),
        "change_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "required_speedup": 1.1,
    },
)

FROZEN_CONFIG_ANTIGRAVITY = Exp471ConfigAntigravity()


__all__ = ("Exp471ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
