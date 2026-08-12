"""Finite-sample, label-free calibration for paper score-source evidence."""
from __future__ import annotations

from numbers import Real

import numpy as np

from .contracts_paper import LeafEvidence


def _numeric_vector(values: object, name: str, *, nonempty: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if (
        array.ndim != 1
        or (nonempty and array.size == 0)
        or array.dtype.kind not in "iuf"
    ):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}finite numeric vector")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite numeric vector")
    return result


def _boolean_mask(values: object, size: int, name: str = "mask") -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.shape[0] != size or array.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be an aligned bool mask")
    return array


def _immutable(values: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validated_alpha(alpha: object) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite real number")
    value = float(alpha)
    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("alpha must be finite and in (0, 1]")
    return value


def upper_tail_pvalues(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return smoothed empirical upper-tail p-values for finite score vectors."""
    reference_values = _numeric_vector(reference, "reference", nonempty=True)
    query_values = _numeric_vector(query, "query")
    return np.asarray(
        [
            (1 + np.count_nonzero(reference_values >= value)) / (reference_values.size + 1)
            for value in query_values
        ],
        dtype=np.float64,
    )


def leaf_evidence(reference: np.ndarray, query: np.ndarray, alpha: float) -> LeafEvidence:
    """Build immutable continuous evidence without changing the native p-value rule."""
    alpha_value = _validated_alpha(alpha)
    pvalue = upper_tail_pvalues(reference, query)
    return LeafEvidence(
        pvalue=_immutable(pvalue, dtype=np.dtype(np.float64)),
        margin=_immutable(np.log10(alpha_value / pvalue), dtype=np.dtype(np.float64)),
        strength=_immutable(1.0 - pvalue, dtype=np.dtype(np.float64)),
        native_pass=_immutable(pvalue <= alpha_value, dtype=np.dtype(bool)),
        evidence_missing=_immutable(np.zeros(pvalue.shape, dtype=bool), dtype=np.dtype(bool)),
    )


def mark_missing_evidence(evidence: LeafEvidence, missing_mask: np.ndarray) -> LeafEvidence:
    """Explicitly replace missing leaves with the neutral evidence sentinel."""
    if not isinstance(evidence, LeafEvidence):
        raise TypeError("evidence must be LeafEvidence")
    pvalue = _numeric_vector(evidence.pvalue, "pvalue")
    margin = _numeric_vector(evidence.margin, "margin")
    strength = _numeric_vector(evidence.strength, "strength")
    native_pass = _boolean_mask(evidence.native_pass, pvalue.size, "native_pass")
    existing_missing = _boolean_mask(evidence.evidence_missing, pvalue.size, "evidence_missing")
    if margin.shape != pvalue.shape or strength.shape != pvalue.shape:
        raise ValueError("evidence fields must be aligned")
    if np.any((pvalue <= 0.0) | (pvalue > 1.0)) or np.any((strength < 0.0) | (strength >= 1.0)):
        raise ValueError("evidence values are outside the finite-sample range")
    missing = _boolean_mask(missing_mask, pvalue.size)
    combined_missing = existing_missing | missing
    return LeafEvidence(
        pvalue=_immutable(np.where(combined_missing, 0.5, pvalue), dtype=np.dtype(np.float64)),
        margin=_immutable(np.where(combined_missing, 0.0, margin), dtype=np.dtype(np.float64)),
        strength=_immutable(np.where(combined_missing, 0.5, strength), dtype=np.dtype(np.float64)),
        native_pass=_immutable(np.where(combined_missing, False, native_pass), dtype=np.dtype(bool)),
        evidence_missing=_immutable(combined_missing, dtype=np.dtype(bool)),
    )


__all__ = ["leaf_evidence", "mark_missing_evidence", "upper_tail_pvalues"]
