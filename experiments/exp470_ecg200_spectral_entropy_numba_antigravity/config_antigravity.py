"""Frozen Exp470 spectral-entropy authority; scoring is deliberately out of scope."""

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
class Exp470ConfigAntigravity:
    seeds: tuple[int, int] = (20260721, 20261721)
    folds: int = 5
    normal_rows: int = 69
    source_width: int = 96
    interpolated_width: int = 140
    onset: int = 56
    window_length: int = 28
    window_starts: tuple[int, ...] = (0, 14, 28, 42, 56)
    dft_bins: int = 14
    power_epsilon: float = 1e-12
    method_order: tuple[str, str, str] = ("C0", "E11", "C11")
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
                raise ValueError(f"{name} differs from the frozen Exp470 configuration")


_FROZEN_VALUES_ANTIGRAVITY = MappingProxyType(
    {
        "seeds": (20260721, 20261721),
        "folds": 5,
        "normal_rows": 69,
        "source_width": 96,
        "interpolated_width": 140,
        "onset": 56,
        "window_length": 28,
        "window_starts": (0, 14, 28, 42, 56),
        "dft_bins": 14,
        "power_epsilon": 1e-12,
        "method_order": ("C0", "E11", "C11"),
        "change_types": (
            "level_shift",
            "gradual_drift",
            "amplitude_scaling",
            "period_change",
        ),
        "required_speedup": 1.1,
    },
)

FROZEN_CONFIG_ANTIGRAVITY = Exp470ConfigAntigravity()


__all__ = ("Exp470ConfigAntigravity", "FROZEN_CONFIG_ANTIGRAVITY")
