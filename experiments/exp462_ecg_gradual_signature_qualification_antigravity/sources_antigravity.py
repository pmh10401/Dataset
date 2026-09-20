"""Replay the exact Exp461 normal_2 TRAIN matrix for Exp462."""

from __future__ import annotations

import numpy as np

from experiments.exp461_ecg_period_signature_qualification_antigravity.sources_antigravity import (
    load_normal2_train_antigravity as _load_exp461_train_antigravity,
)


def load_normal2_train_antigravity() -> np.ndarray:
    matrix = np.array(
        _load_exp461_train_antigravity(),
        dtype=np.float32,
        copy=True,
        order="C",
    )
    if matrix.shape != (1413, 140):
        raise ValueError("TRAIN matrix shape drifted from the frozen contract")
    if matrix.dtype != np.dtype(np.float32):
        raise ValueError("TRAIN matrix dtype drifted from the frozen contract")
    if not np.isfinite(matrix).all():
        raise ValueError("TRAIN matrix must remain finite")
    matrix.setflags(write=False)
    return matrix


__all__ = ("load_normal2_train_antigravity",)
