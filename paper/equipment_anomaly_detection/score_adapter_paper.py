"""Deterministic adapters from evidence strength to frozen alert routes."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _finite_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite numeric vector")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite numeric vector")
    return result


def _strength_vector(values: object) -> np.ndarray:
    strength = _finite_vector(values, "strength")
    if np.any((strength < 0.0) | (strength >= 1.0)):
        raise ValueError("strength must be in [0, 1)")
    return strength


def _bool_vector(values: object, name: str, size: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size or array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be an aligned bool vector")
    return array


def strict_native_pass(raw_scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Preserve a native strict raw-score boundary without p-value substitution."""
    scores = _finite_vector(raw_scores, "raw_scores")
    boundary = _finite_vector(thresholds, "thresholds")
    if scores.shape != boundary.shape:
        raise ValueError("raw_scores and thresholds must be aligned")
    return scores > boundary


def combine_evidence(values: Sequence[np.ndarray], mode: str, k: int = 1) -> np.ndarray:
    """Aggregate aligned leaf strengths using the frozen Boolean rule's order statistic."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError("values must be a nonempty sequence")
    vectors = tuple(_strength_vector(value) for value in values)
    if len({vector.shape[0] for vector in vectors}) != 1:
        raise ValueError("evidence values must be aligned")
    matrix = np.vstack(vectors)
    if mode == "and":
        return np.min(matrix, axis=0)
    if mode == "or":
        return np.max(matrix, axis=0)
    if mode == "k_of_n":
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= int(k) <= matrix.shape[0]:
            raise ValueError("k must be an integer from 1 through the evidence count")
        return np.sort(matrix, axis=0)[-int(k)]
    raise ValueError("unknown evidence mode")


def route_preserving_scores(
    strength: np.ndarray,
    alerts: np.ndarray,
    evidence_missing: np.ndarray | None = None,
) -> np.ndarray:
    """Map strengths into alert/non-alert bands without changing frozen route decisions."""
    values = _strength_vector(strength)
    routes = _bool_vector(alerts, "alerts", values.size)
    if evidence_missing is not None:
        missing = _bool_vector(evidence_missing, "evidence_missing mask", values.size)
        values = np.where(missing, 0.5, values)
    return np.where(routes, 0.5 + 0.5 * values, 0.5 * values)


__all__ = ["combine_evidence", "route_preserving_scores", "strict_native_pass"]
