"""Encode and replay raw evidence vectors for Exp464 Task 4."""

from __future__ import annotations

import base64
from math import isfinite
from pathlib import Path
import zlib

import numpy as np

from .centered_ramp_antigravity import centered_suffix_basis_antigravity
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


RAW_CENTERED_ANTIGRAVITY = "raw_centered"
TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY = "template_residual_centered"
TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY = "template_residual_uncentered"
_VECTOR_WIDTH_ANTIGRAVITY = (
    FROZEN_CONFIG_ANTIGRAVITY.normal2_width - FROZEN_CONFIG_ANTIGRAVITY.onset
)


def _vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.shape != (FROZEN_CONFIG_ANTIGRAVITY.normal2_width,):
        raise ValueError(f"{name} must be a one-dimensional 140-wide ndarray")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _matrix_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 2 or values.shape[1] != _VECTOR_WIDTH_ANTIGRAVITY:
        raise ValueError(f"{name} must be a two-dimensional suffix matrix")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{name} must use a floating dtype")
    result = np.asarray(values, dtype=np.float64, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def raw_centered_vector_antigravity(values: np.ndarray) -> np.ndarray:
    row = _vector_antigravity(values, "values")
    suffix = row[FROZEN_CONFIG_ANTIGRAVITY.onset :]
    centered = suffix - float(np.mean(suffix))
    return np.asarray(centered, dtype=np.float64, order="C")


def template_residual_centered_vector_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    row = _vector_antigravity(values, "values")
    reference = _vector_antigravity(template, "template")
    residual = row - reference
    suffix = residual[FROZEN_CONFIG_ANTIGRAVITY.onset :]
    centered = suffix - float(np.mean(suffix))
    return np.asarray(centered, dtype=np.float64, order="C")


def template_residual_uncentered_vector_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    row = _vector_antigravity(values, "values")
    reference = _vector_antigravity(template, "template")
    residual = row - reference
    suffix = residual[FROZEN_CONFIG_ANTIGRAVITY.onset :]
    return np.asarray(suffix, dtype=np.float64, order="C")


def encode_vector_matrix_antigravity(values: np.ndarray) -> str:
    matrix = _matrix_antigravity(values, "values")
    compressed = zlib.compress(matrix.tobytes(order="C"), level=9)
    return f"{matrix.shape[0]}:{matrix.shape[1]}:{base64.b64encode(compressed).decode('ascii')}"


def decode_vector_matrix_antigravity(blob: object, name: str) -> np.ndarray:
    if type(blob) is not str or not blob:
        raise ValueError(f"{name} must be a non-empty encoded matrix blob")
    parts = blob.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"{name} blob is invalid")
    rows_text, width_text, payload = parts
    try:
        rows = int(rows_text)
        width = int(width_text)
    except ValueError as error:
        raise ValueError(f"{name} dimensions are invalid") from error
    if rows < 1 or width != _VECTOR_WIDTH_ANTIGRAVITY:
        raise ValueError(f"{name} dimensions are invalid")
    try:
        compressed = base64.b64decode(payload.encode("ascii"), validate=True)
        raw = zlib.decompress(compressed)
    except (ValueError, zlib.error) as error:
        raise ValueError(f"{name} payload is invalid") from error
    matrix = np.frombuffer(raw, dtype=np.float64)
    if matrix.size != rows * width:
        raise ValueError(f"{name} decoded size is invalid")
    result = np.asarray(matrix.reshape(rows, width), dtype=np.float64, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} decoded matrix is non-finite")
    return result


def amplitudes_cosines_from_vectors_antigravity(
    kind: str,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _matrix_antigravity(values, "values")
    if kind in (RAW_CENTERED_ANTIGRAVITY, TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY):
        basis = centered_suffix_basis_antigravity()
    elif kind == TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY:
        t = np.linspace(0.0, 1.0, matrix.shape[1], dtype=np.float64)
        basis = t / float(np.linalg.norm(t))
    else:
        raise ValueError("raw evidence kind is invalid")
    amplitudes = np.asarray(matrix @ basis, dtype=np.float64, order="C")
    norms = np.linalg.norm(matrix, axis=1)
    cosines = np.asarray(
        np.where(norms == 0.0, 0.0, amplitudes / norms),
        dtype=np.float64,
        order="C",
    )
    if not np.isfinite(amplitudes).all() or not np.isfinite(cosines).all():
        raise ValueError("decoded raw evidence is non-finite")
    return amplitudes, cosines


__all__ = (
    "RAW_CENTERED_ANTIGRAVITY",
    "TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY",
    "TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY",
    "amplitudes_cosines_from_vectors_antigravity",
    "decode_vector_matrix_antigravity",
    "encode_vector_matrix_antigravity",
    "raw_centered_vector_antigravity",
    "template_residual_centered_vector_antigravity",
    "template_residual_uncentered_vector_antigravity",
)
