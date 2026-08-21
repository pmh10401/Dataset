"""Replay the exact Exp462 normal_2 TRAIN matrix for Exp463."""

from __future__ import annotations

import numpy as np

from experiments.exp462_ecg_gradual_signature_qualification_antigravity.sources_antigravity import (
    load_normal2_train_antigravity as _load_exp462_train_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


TRAIN_LABEL_QUERY_COUNT_ANTIGRAVITY = 0


def load_normal2_train_antigravity() -> np.ndarray:
    """Return an owned, finite, read-only float32 copy of the TRAIN matrix."""

    matrix = np.array(
        _load_exp462_train_antigravity(),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    expected_shape = (
        FROZEN_CONFIG_ANTIGRAVITY.normal2_train_rows,
        FROZEN_CONFIG_ANTIGRAVITY.normal2_width,
    )
    if matrix.shape != expected_shape:
        raise ValueError("TRAIN matrix shape drifted from the frozen contract")
    if matrix.dtype != np.dtype(np.float32):
        raise ValueError("TRAIN matrix dtype drifted from the frozen contract")
    if not np.isfinite(matrix).all():
        raise ValueError("TRAIN matrix must remain finite")
    matrix.setflags(write=False)
    return matrix


__all__ = ("load_normal2_train_antigravity",)
