"""Deterministic prefix-scaled ramp features for Exp463 Task 3."""

from __future__ import annotations

import numpy as np


_ROW_WIDTH_ANTIGRAVITY = 140
_ONSET_ANTIGRAVITY = 56
_MAD_TO_NORMAL_SCALE_ANTIGRAVITY = 1.4826


def _input_row_antigravity(values: object, name: str = "values") -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional ndarray")
    if values.shape != (_ROW_WIDTH_ANTIGRAVITY,):
        raise ValueError(f"{name} must have length {_ROW_WIDTH_ANTIGRAVITY}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    row = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(row).all():
        raise ValueError(f"{name} must be finite")
    return row


def _ramp_basis_antigravity() -> np.ndarray:
    basis = np.zeros(_ROW_WIDTH_ANTIGRAVITY, dtype=np.float64)
    basis[_ONSET_ANTIGRAVITY:] = np.linspace(
        0.0,
        1.0,
        _ROW_WIDTH_ANTIGRAVITY - _ONSET_ANTIGRAVITY,
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(basis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("ramp basis norm must be positive and finite")
    result = basis / norm
    result.setflags(write=False)
    return result


def _prefix_center_scale_antigravity(row: np.ndarray) -> tuple[float, float]:
    prefix = row[0:_ONSET_ANTIGRAVITY]
    center = float(np.median(prefix))
    scale = float(
        _MAD_TO_NORMAL_SCALE_ANTIGRAVITY
        * np.median(np.abs(prefix - center))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("prefix robust scale must be positive and finite")
    return center, scale


def _ramp_score_from_prefix_antigravity(row: np.ndarray) -> float:
    center, scale = _prefix_center_scale_antigravity(row)
    normalized = (row - center) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("prefix-normalized row must be finite")
    score = float(np.dot(normalized, _ramp_basis_antigravity()))
    if not np.isfinite(score):
        raise ValueError("prefix ramp score must be finite")
    return score


def prefix_ramp_score_antigravity(values: np.ndarray) -> float:
    """Return the signed prefix-normalized ramp dot product for one row."""

    return _ramp_score_from_prefix_antigravity(_input_row_antigravity(values))


def template_residual_ramp_score_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> float:
    """Score a row after subtracting a TRAIN-fit timewise median template."""

    row = _input_row_antigravity(values)
    reference = _input_row_antigravity(template, "template")
    return _ramp_score_from_prefix_antigravity(row - reference)


def whole_series_ramp_score_antigravity(values: np.ndarray) -> float:
    """Return a ramp score normalized by the row's whole-series median/MAD."""

    row = _input_row_antigravity(values)
    center = float(np.median(row))
    scale = float(
        _MAD_TO_NORMAL_SCALE_ANTIGRAVITY * np.median(np.abs(row - center))
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("whole-series robust scale must be positive and finite")
    normalized = (row - center) / scale
    score = float(np.dot(normalized, _ramp_basis_antigravity()))
    if not np.isfinite(score):
        raise ValueError("whole-series ramp score must be finite")
    return score


__all__ = ("prefix_ramp_score_antigravity",)
