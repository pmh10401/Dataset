from __future__ import annotations

import numpy as np


def empirical_pvalues(train_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    reference = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    query = np.asarray(query_scores, dtype=np.float64).reshape(-1)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("finite non-empty TRAIN OOF scores required")
    if not np.isfinite(query).all():
        raise ValueError("finite query scores required")
    return np.asarray(
        [(1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0) for value in query],
        dtype=np.float64,
    )


def candidate_indices(pvalues: np.ndarray, alpha: float, *, n_reference: int | None = None) -> list[int]:
    values = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    if n_reference is not None and float(alpha) + 1e-15 < 1.0 / (int(n_reference) + 1.0):
        return []
    return np.flatnonzero(values <= float(alpha)).astype(int).tolist()


def calibration_resolution(n_reference: int, alpha: float) -> dict[str, object]:
    attainable = 1.0 / (int(n_reference) + 1.0)
    return {
        "attainable_p": attainable,
        "resolution_blocked": float(alpha) + 1e-15 < attainable,
    }
