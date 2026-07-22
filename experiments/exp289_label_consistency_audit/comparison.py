from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


NORMALIZED_EQUIVALENT_RMSE = 1e-6
NEAR_DUPLICATE_RMSE = 0.01
NEAR_DUPLICATE_CORRELATION_DISTANCE = 1e-4
CONSTANT_EPSILON = 1e-12


@dataclass(frozen=True)
class PairComparison:
    tier: str
    raw_exact: bool
    aligned_exact: bool
    normalized_rmse: float
    correlation_distance: float
    aligned_rmse: float
    aligned_max_abs: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _row(values: np.ndarray) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(row) == 0 or not np.isfinite(row).all():
        raise ValueError("comparison rows must be non-empty and finite")
    return row


def _normalize(row: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(row, dtype=np.float64)
    scale = float(values.std())
    if scale <= CONSTANT_EPSILON:
        return np.zeros_like(values), True
    return (values - float(values.mean())) / scale, False


def compare_pair(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    left_aligned: np.ndarray,
    right_aligned: np.ndarray,
) -> PairComparison:
    left_raw_row = _row(left_raw)
    right_raw_row = _row(right_raw)
    left = _row(left_aligned)
    right = _row(right_aligned)
    if left.shape != right.shape:
        raise ValueError("aligned rows must have the same shape")

    raw_exact = left_raw_row.shape == right_raw_row.shape and bool(
        np.array_equal(left_raw_row, right_raw_row)
    )
    aligned_exact = bool(np.array_equal(left, right))
    difference = left.astype(np.float64) - right.astype(np.float64)
    aligned_rmse = float(np.sqrt(np.mean(difference * difference)))
    aligned_max_abs = float(np.max(np.abs(difference)))

    left_normalized, left_constant = _normalize(left)
    right_normalized, right_constant = _normalize(right)
    if left_constant or right_constant:
        normalized_rmse = 0.0 if aligned_exact else float("inf")
        correlation_distance = 0.0 if aligned_exact else 1.0
    else:
        normalized_difference = left_normalized - right_normalized
        normalized_rmse = float(
            np.sqrt(np.mean(normalized_difference * normalized_difference))
        )
        correlation = float(np.mean(left_normalized * right_normalized))
        correlation_distance = float(np.clip(1.0 - correlation, 0.0, 2.0))

    if raw_exact:
        tier = "raw_exact"
    elif aligned_exact:
        tier = "aligned_exact"
    elif normalized_rmse <= NORMALIZED_EQUIVALENT_RMSE:
        tier = "normalized_equivalent"
    elif (
        normalized_rmse <= NEAR_DUPLICATE_RMSE
        and correlation_distance <= NEAR_DUPLICATE_CORRELATION_DISTANCE
    ):
        tier = "near_duplicate"
    else:
        tier = "distinct"

    return PairComparison(
        tier=tier,
        raw_exact=raw_exact,
        aligned_exact=aligned_exact,
        normalized_rmse=normalized_rmse,
        correlation_distance=correlation_distance,
        aligned_rmse=aligned_rmse,
        aligned_max_abs=aligned_max_abs,
    )


__all__ = [
    "CONSTANT_EPSILON",
    "NEAR_DUPLICATE_CORRELATION_DISTANCE",
    "NEAR_DUPLICATE_RMSE",
    "NORMALIZED_EQUIVALENT_RMSE",
    "PairComparison",
    "compare_pair",
]
