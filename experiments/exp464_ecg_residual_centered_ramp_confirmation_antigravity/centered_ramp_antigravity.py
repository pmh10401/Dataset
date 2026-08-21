"""Centered residual-ramp evidence for Exp464 Task 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_ROW_WIDTH_ANTIGRAVITY = 140
_ONSET_ANTIGRAVITY = 56
_SUFFIX_WIDTH_ANTIGRAVITY = _ROW_WIDTH_ANTIGRAVITY - _ONSET_ANTIGRAVITY


def _input_row_antigravity(values: object, name: str) -> np.ndarray:
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


def _scoring_basis_antigravity() -> np.ndarray:
    t = np.linspace(0.0, 1.0, _SUFFIX_WIDTH_ANTIGRAVITY, dtype=np.float64)
    centered = t - float(np.mean(t))
    norm = float(np.linalg.norm(centered))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("centered suffix basis norm must be positive and finite")
    return centered / norm


def centered_suffix_basis_antigravity() -> np.ndarray:
    basis = np.array(_scoring_basis_antigravity(), copy=True, order="C")
    basis[-1] -= float(np.sum(basis, dtype=np.float64))
    basis.setflags(write=False)
    return basis


@dataclass(frozen=True, slots=True)
class CenteredRampEvidenceAntigravity:
    amplitude: float
    cosine: float


def centered_ramp_evidence_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> CenteredRampEvidenceAntigravity:
    row = _input_row_antigravity(values, "values")
    reference = _input_row_antigravity(template, "template")
    residual = row - reference
    suffix = residual[_ONSET_ANTIGRAVITY:]
    centered = suffix - float(np.mean(suffix))
    amplitude = float(np.dot(centered, _scoring_basis_antigravity()))
    norm = float(np.linalg.norm(centered))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    if np.isclose(cosine, 1.0, rtol=0.0, atol=1e-15):
        cosine = 1.0
    elif np.isclose(cosine, -1.0, rtol=0.0, atol=1e-15):
        cosine = -1.0
    elif np.isclose(cosine, 0.0, rtol=0.0, atol=1e-15):
        cosine = 0.0
    return CenteredRampEvidenceAntigravity(amplitude=amplitude, cosine=cosine)


__all__ = (
    "CenteredRampEvidenceAntigravity",
    "centered_ramp_evidence_antigravity",
    "centered_suffix_basis_antigravity",
)
