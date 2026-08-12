"""TRAIN-reference calibration and deterministic alert-budget rules."""

from __future__ import annotations

import numpy as np


def upper_tail_p_values(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Finite-sample upper-tail conformal p-values for anomaly-up scores."""
    reference_values = _finite_vector(reference, "reference")
    query_values = _finite_vector(query, "query")
    return np.asarray(
        [
            (1 + np.count_nonzero(reference_values >= value))
            / (len(reference_values) + 1)
            for value in query_values
        ],
        dtype=np.float64,
    )


def conformal_alerts(
    reference: np.ndarray,
    query: np.ndarray,
    test_indices: tuple[int, ...],
    alpha: float,
) -> tuple[int, ...]:
    """Return TEST indices whose upper-tail p-value is at most alpha."""
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    p_values = upper_tail_p_values(reference, query)
    if len(test_indices) != len(p_values) or len(set(test_indices)) != len(test_indices):
        raise ValueError("test indices must be unique and aligned with query scores")
    return tuple(
        int(index)
        for index, p_value in zip(test_indices, p_values, strict=True)
        if p_value <= float(alpha)
    )


def matched_top_k(
    scores: np.ndarray, indices: tuple[int, ...], k: int
) -> tuple[int, ...]:
    """Select anomaly-up top-k scores with TEST index as the tie breaker."""
    values = _finite_vector(scores, "scores")
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError("indices must be unique and aligned with scores")
    if type(k) is not int or not 0 <= k <= len(values):
        raise ValueError("k must be an integer within the score population")
    order = sorted(
        range(len(values)),
        key=lambda position: (-float(values[position]), int(indices[position])),
    )
    return tuple(sorted(int(indices[position]) for position in order[:k]))


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result
