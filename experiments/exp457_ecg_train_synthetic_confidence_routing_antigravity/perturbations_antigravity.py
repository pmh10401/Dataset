"""Exp417-compatible deterministic primary perturbations for finite TRAIN rows."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _readonly_antigravity(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


def build_primary_perturbations_antigravity(
    values: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Return the four frozen primary perturbations without mutating ``values``."""
    if type(values) is not np.ndarray or values.ndim != 1:
        raise ValueError("values must be a one-dimensional numpy array")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("values must have a floating dtype")
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("values must contain at least two finite points")

    series = np.array(values, dtype=np.float64, copy=True)
    center = float(np.median(series))
    robust_scale = float(
        _MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(series - center))
    )
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("values must have a positive finite Exp417 robust scale")

    onset = int(np.floor(0.40 * series.size))
    duration = series.size - onset

    level = np.array(series, copy=True)
    level[onset:] += 3.0 * robust_scale

    drift = np.array(series, copy=True)
    drift[onset:] += np.linspace(
        0.0, 3.0 * robust_scale, duration, dtype=np.float64
    )

    amplitude = center + 1.5 * (series - center)

    warped = np.array(series, copy=True)
    region = series[onset:]
    source = np.arange(duration, dtype=np.float64)
    warped[onset:] = np.interp(source / 1.25, source, region)

    perturbations = (
        _readonly_antigravity(level),
        _readonly_antigravity(drift),
        _readonly_antigravity(amplitude),
        _readonly_antigravity(warped),
    )
    return MappingProxyType(
        dict(zip(FROZEN_CONFIG_ANTIGRAVITY.primary_types, perturbations, strict=True))
    )


__all__ = ("build_primary_perturbations_antigravity",)
