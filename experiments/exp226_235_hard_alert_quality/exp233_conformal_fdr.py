from __future__ import annotations

from collections.abc import Iterable

import numpy as np


FDR_LEVELS = (0.005, 0.01, 0.02, 0.05)


def bh_indices(pvalues: Iterable[float], *, q: float) -> set[int]:
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("p-values must be a finite vector")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be in [0, 1]")
    if not 0.0 < float(q) < 1.0:
        raise ValueError("q must be between zero and one")
    count = len(values)
    if count == 0:
        return set()
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    thresholds = float(q) * np.arange(1, count + 1) / count
    valid = np.flatnonzero(ordered <= thresholds)
    if len(valid) == 0:
        return set()
    cutoff = ordered[int(valid[-1])]
    return set(np.flatnonzero(values <= cutoff).astype(int))
