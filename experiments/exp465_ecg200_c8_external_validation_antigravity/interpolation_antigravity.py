"""Fixed 96-to-140 linear interpolation for Exp465."""

from __future__ import annotations

import numpy as np


_SOURCE_WIDTH_ANTIGRAVITY = 96
_TARGET_WIDTH_ANTIGRAVITY = 140
_SOURCE_GRID_ANTIGRAVITY = np.linspace(0.0, 1.0, _SOURCE_WIDTH_ANTIGRAVITY, dtype=np.float64)
_TARGET_GRID_ANTIGRAVITY = np.linspace(0.0, 1.0, _TARGET_WIDTH_ANTIGRAVITY, dtype=np.float64)


def interpolate_to_140_antigravity(values: np.ndarray) -> np.ndarray:
    """Apply the frozen linear 96-to-140 interpolation once per row."""
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be a numpy ndarray")
    if values.ndim != 2 or not isinstance(values.ndim, int):
        raise ValueError("values must be exactly 2-D")
    if values.dtype is not np.dtype(np.float64):
        raise ValueError(f"values must be exact float64, got {values.dtype}")
    if values.shape[1] != _SOURCE_WIDTH_ANTIGRAVITY:
        raise ValueError(f"values width must be {_SOURCE_WIDTH_ANTIGRAVITY}, got {values.shape[1]}")
    if values.shape[0] == 0:
        raise ValueError("values must be nonempty")
    if not np.isfinite(values).all():
        raise ValueError("values must be finite")

    result = np.array(
        [np.interp(_TARGET_GRID_ANTIGRAVITY, _SOURCE_GRID_ANTIGRAVITY, row) for row in values],
        dtype=np.float64,
        copy=True,
        order="C",
    )
    result.setflags(write=False)
    return result


__all__ = ("interpolate_to_140_antigravity",)
