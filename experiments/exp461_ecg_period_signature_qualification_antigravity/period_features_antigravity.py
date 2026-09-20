"""Deterministic 120D period signatures for Exp461 Task 3."""

from __future__ import annotations

import numpy as np


_ROW_WIDTH_ANTIGRAVITY = 140
_BAND_COUNT_ANTIGRAVITY = 16
_LAG_COUNT_ANTIGRAVITY = 24
_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _input_row_antigravity(values: np.ndarray) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1:
        raise ValueError("values must be a one-dimensional ndarray")
    if values.shape != (_ROW_WIDTH_ANTIGRAVITY,):
        raise ValueError("values must have length 140")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError("values must use a floating dtype")
    row = np.asarray(values, dtype=np.float64)
    if not np.isfinite(row).all():
        raise ValueError("values must be finite")
    return row


def _view_signature_antigravity(view: np.ndarray) -> np.ndarray:
    centered = np.asarray(view, dtype=np.float64) - float(np.median(view))
    scale = float(_MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("view robust scale must be positive and finite")
    z = centered / scale
    if not np.isfinite(z).all():
        raise ValueError("standardized view must be finite")

    power = np.abs(np.fft.rfft(z * np.hanning(len(z)))) ** 2
    positive = np.asarray(power[1:], dtype=np.float64)
    total_power = float(np.sum(positive))
    if not np.isfinite(total_power) or total_power <= 0.0:
        raise ValueError("positive-frequency power must be positive and finite")
    bands = np.asarray(
        [float(np.sum(group) / total_power) for group in np.array_split(positive, _BAND_COUNT_ANTIGRAVITY)],
        dtype=np.float64,
    )

    denominator = float(np.dot(z, z))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("autocorrelation denominator must be positive and finite")
    autocorrelation = np.asarray(
        [
            float(np.dot(z[:-lag], z[lag:]) / denominator)
            for lag in range(1, _LAG_COUNT_ANTIGRAVITY + 1)
        ],
        dtype=np.float64,
    )
    view_result = np.concatenate((bands, autocorrelation)).astype(np.float64, copy=False)
    if view_result.shape != (40,) or not np.isfinite(view_result).all():
        raise ValueError("view signature must be finite length 40")
    return view_result


def period_signature_antigravity(values: np.ndarray) -> np.ndarray:
    row = _input_row_antigravity(values)
    result = np.concatenate(
        (
            _view_signature_antigravity(row[0:140]),
            _view_signature_antigravity(row[0:70]),
            _view_signature_antigravity(row[70:140]),
        )
    )
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    if result.shape != (120,) or not np.isfinite(result).all():
        raise ValueError("period signature must be finite length 120")
    result.setflags(write=False)
    return result


__all__ = ("period_signature_antigravity",)
