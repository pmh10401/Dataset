"""Chunked Mann-Whitney-style AUC for Exp458 Task 2."""

from __future__ import annotations

import numpy as np


_CHUNK_ROWS_ANTIGRAVITY = 4096


def _exact_finite_real_vector_antigravity(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.ndim != 1 or value.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if value.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must contain exact real scalars")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def mann_whitney_auc_antigravity(normal: np.ndarray, anomaly: np.ndarray) -> float:
    normal_vector = _exact_finite_real_vector_antigravity(normal, "normal")
    anomaly_vector = _exact_finite_real_vector_antigravity(anomaly, "anomaly")

    wins = 0
    ties = 0
    for start in range(0, anomaly_vector.size, _CHUNK_ROWS_ANTIGRAVITY):
        stop = min(start + _CHUNK_ROWS_ANTIGRAVITY, anomaly_vector.size)
        block = anomaly_vector[start:stop, None]
        wins += int(np.count_nonzero(block > normal_vector[None, :]))
        ties += int(np.count_nonzero(block == normal_vector[None, :]))
    total_pairs = normal_vector.size * anomaly_vector.size
    return float((wins + 0.5 * ties) / total_pairs)


__all__ = ("mann_whitney_auc_antigravity",)
