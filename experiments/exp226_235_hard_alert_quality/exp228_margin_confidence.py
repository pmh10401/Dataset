from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def normalized_score_margin(scores: Iterable[float], threshold: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    scale = max(abs(float(threshold)), 1e-12)
    return (values - float(threshold)) / scale


def conformal_margin(pvalues: Iterable[float], alpha: float) -> np.ndarray:
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite vector")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    return (float(alpha) - values) / float(alpha)

