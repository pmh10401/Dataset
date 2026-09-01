"""Optional Numba kernels for Exp481's exact frozen reference operations."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _nearest_distinct_family_indices_kernel_antigravity(
    distances: np.ndarray, family_codes: np.ndarray, k: int
) -> np.ndarray:
    size = len(distances)
    order = np.arange(size, dtype=np.int64)
    # Stable insertion sort makes equal-distance behavior explicit and portable.
    for position in range(1, size):
        index = order[position]
        value = distances[index]
        cursor = position
        while cursor > 0 and distances[order[cursor - 1]] > value:
            order[cursor] = order[cursor - 1]
            cursor -= 1
        order[cursor] = index
    chosen = np.empty(min(k, size), dtype=np.int64)
    seen = np.zeros(int(np.max(family_codes)) + 1, dtype=np.uint8)
    count = 0
    for index in order:
        family = family_codes[index]
        if seen[family] == 0:
            seen[family] = 1
            chosen[count] = index
            count += 1
            if count == k:
                break
    return chosen[:count]


@njit(cache=True)
def _strict_statistics_kernel_antigravity(
    zf_gains: np.ndarray, auc_gains: np.ndarray, q: float
) -> tuple[bool, int, int, int, float, float]:
    wins = 0
    losses = 0
    for gain in zf_gains:
        if gain > 0:
            wins += 1
        elif gain < 0:
            losses += 1
    ties = len(zf_gains) - wins - losses
    ordered = np.sort(auc_gains)
    position = q * (len(ordered) - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    weight = position - lower
    quantile = float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)
    minimum = float(ordered[0])
    passed = wins > losses and minimum >= 0.0 and quantile > 0.0
    return passed, wins, losses, ties, minimum, quantile


def nearest_distinct_family_indices_numba_antigravity(
    distances: np.ndarray, family_codes: np.ndarray, k: int
) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    codes = np.asarray(family_codes, dtype=np.int64)
    if values.ndim != 1 or codes.ndim != 1 or len(values) != len(codes) or not len(values):
        raise ValueError("distances and family_codes must be aligned non-empty vectors")
    if not np.isfinite(values).all() or np.any(codes < 0):
        raise ValueError("distances must be finite and family codes nonnegative")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    return _nearest_distinct_family_indices_kernel_antigravity(values, codes, k)


def strict_statistics_numba_antigravity(
    zf_gains: np.ndarray, auc_gains: np.ndarray, q: float
) -> tuple[bool, int, int, int, float, float]:
    zf = np.asarray(zf_gains, dtype=np.int64)
    auc = np.asarray(auc_gains, dtype=np.float64)
    q_value = float(q)
    if zf.ndim != 1 or auc.ndim != 1 or len(zf) != len(auc) or not len(zf):
        raise ValueError("gain arrays must be aligned non-empty vectors")
    if not np.isfinite(auc).all() or not np.isfinite(q_value) or not 0.0 <= q_value <= 1.0:
        raise ValueError("AUC gains and q must be finite, with q in [0, 1]")
    result = _strict_statistics_kernel_antigravity(zf, auc, q_value)
    return (
        bool(result[0]), int(result[1]), int(result[2]), int(result[3]),
        float(result[4]), float(result[5]),
    )


__all__ = [
    "nearest_distinct_family_indices_numba_antigravity",
    "strict_statistics_numba_antigravity",
]
