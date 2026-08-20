"""Deterministic 24D gradual signatures for Exp462 Task 3."""

from __future__ import annotations

import numpy as np


_ROW_WIDTH_ANTIGRAVITY = 140
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


def _normalized_view_antigravity(view: np.ndarray) -> np.ndarray:
    centered = np.asarray(view, dtype=np.float64) - float(np.median(view))
    scale = float(_MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(centered)))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("view robust scale must be positive and finite")
    normalized = centered / scale
    if not np.isfinite(normalized).all():
        raise ValueError("normalized view must be finite")
    return normalized


def _basis_values_antigravity(size: int) -> tuple[np.ndarray, ...]:
    u = np.linspace(0.0, 1.0, size, dtype=np.float64)
    return (
        u,
        u * u,
        np.maximum(u - 0.25, 0.0),
        np.maximum(u - 0.50, 0.0),
        np.maximum(u - 0.75, 0.0),
    )


def _view_signature_antigravity(view: np.ndarray) -> np.ndarray:
    normalized = _normalized_view_antigravity(view)
    values: list[float] = []
    for basis in _basis_values_antigravity(len(normalized)):
        centered_basis = basis - float(np.mean(basis))
        norm = float(np.linalg.norm(centered_basis))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("basis norm must be positive and finite")
        values.append(float(np.dot(normalized, centered_basis / norm)))
    quarters = np.array_split(normalized, 4)
    medians = tuple(float(np.median(part)) for part in quarters)
    values.extend(
        (
            medians[1] - medians[0],
            medians[2] - medians[1],
            medians[3] - medians[2],
        )
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError("view gradual signature must be finite length 8")
    return result


def gradual_signature_antigravity(values: np.ndarray) -> np.ndarray:
    row = _input_row_antigravity(values)
    result = np.concatenate(
        (
            _view_signature_antigravity(row[0:140]),
            _view_signature_antigravity(row[0:70]),
            _view_signature_antigravity(row[70:140]),
        )
    )
    result = np.array(result, dtype=np.float64, copy=True, order="C")
    if result.shape != (24,) or not np.isfinite(result).all():
        raise ValueError("gradual signature must be finite length 24")
    result.setflags(write=False)
    return result


__all__ = ("gradual_signature_antigravity",)
