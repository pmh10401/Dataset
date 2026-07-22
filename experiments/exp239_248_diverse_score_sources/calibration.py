from __future__ import annotations

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import CalibrationResult


def empirical_upper_pvalues(
    train_oof: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    reference = np.asarray(train_oof, dtype=float).reshape(-1)
    query = np.asarray(test_scores, dtype=float).reshape(-1)
    if len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("finite non-empty TRAIN OOF reference required")
    if not np.isfinite(query).all():
        raise ValueError("finite TEST scores required")
    return np.asarray(
        [(1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0) for value in query],
        dtype=float,
    )


def empirical_candidates(
    train_oof: np.ndarray, test_scores: np.ndarray, alpha: float
) -> CalibrationResult:
    reference = np.asarray(train_oof, dtype=float).reshape(-1)
    attainable = 1.0 / (len(reference) + 1.0)
    blocked = float(alpha) + 1e-15 < attainable
    pvalues = empirical_upper_pvalues(reference, test_scores)
    indices = () if blocked else tuple(np.flatnonzero(pvalues <= float(alpha)).astype(int))
    return CalibrationResult(
        pvalues=pvalues,
        indices=indices,
        attainable_p=attainable,
        resolution_blocked=blocked,
    )

