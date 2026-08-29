"""Frozen Exp464 synthetic ECG changes for Exp466."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _readonly_owned_antigravity(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError("synthetic output must be finite")
    result.setflags(write=False)
    return result


def primary_rows_antigravity(row: np.ndarray) -> Mapping[str, np.ndarray]:
    """Return the four frozen Exp464 primary changes for one width-140 row."""
    if not isinstance(row, np.ndarray):
        raise TypeError("row must be a numpy ndarray")
    if row.ndim != 1 or row.shape != (FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,):
        raise ValueError("row must be exactly one-dimensional with width 140")
    if row.dtype != np.dtype(np.float64):
        raise ValueError("row must be exact float64")
    if not np.isfinite(row).all():
        raise ValueError("row must be finite")

    values = np.asarray(row, dtype=np.float64)
    center = float(np.median(values))
    robust_scale = float(
        _MAD_TO_NORMAL_SCALE_ANTIGRAVITY
        * np.median(np.abs(values - center))
    )
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("row robust scale must be positive and finite")

    onset = FROZEN_CONFIG_ANTIGRAVITY.onset
    duration = values.size - onset

    level = np.array(values, dtype=np.float64, copy=True, order="C")
    level[onset:] += 3.0 * robust_scale

    drift = np.array(values, dtype=np.float64, copy=True, order="C")
    drift[onset:] += np.linspace(
        0.0,
        3.0 * robust_scale,
        duration,
        dtype=np.float64,
    )

    amplitude = center + 1.5 * (values - center)

    warped = np.array(values, dtype=np.float64, copy=True, order="C")
    source = np.arange(duration, dtype=np.float64)
    warped[onset:] = np.interp(
        source / FROZEN_CONFIG_ANTIGRAVITY.warp_factor,
        source,
        values[onset:],
    )

    outputs = tuple(
        _readonly_owned_antigravity(candidate)
        for candidate in (level, drift, amplitude, warped)
    )
    return MappingProxyType(
        dict(zip(_CHANGE_TYPES_ANTIGRAVITY, outputs, strict=True)),
    )


__all__ = ("primary_rows_antigravity",)
