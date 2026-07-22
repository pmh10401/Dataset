from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from experiments.exp206_210.core import rankdata_average


def spearman_matrix(
    train_oof_by_source: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    names = sorted(str(name) for name in train_oof_by_source)
    if len(names) < 2:
        raise ValueError("at least two TRAIN OOF sources are required")
    arrays = {name: np.asarray(train_oof_by_source[name], dtype=float) for name in names}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("TRAIN OOF vectors must be aligned one-dimensional arrays")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("TRAIN OOF vectors must be finite")
    ranks = {name: rankdata_average(arrays[name]) for name in names}
    output: dict[str, dict[str, float]] = {name: {} for name in names}
    for left in names:
        for right in names:
            if left == right:
                value = 1.0
            elif np.std(ranks[left]) <= 1e-12 or np.std(ranks[right]) <= 1e-12:
                value = 0.0
            else:
                value = float(np.corrcoef(ranks[left], ranks[right])[0, 1])
            output[left][right] = value
    return output


def correlation_adjusted_weights(
    correlations: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    names = sorted(str(name) for name in correlations)
    if len(names) < 2:
        raise ValueError("at least two sources are required")
    raw: dict[str, float] = {}
    for name in names:
        row = correlations[name]
        if set(row) != set(names):
            raise ValueError("correlation matrix source mismatch")
        redundancy = sum(abs(float(row[other])) for other in names if other != name)
        raw[name] = 1.0 / (1.0 + redundancy)
    total = sum(raw.values())
    return {name: raw[name] / total for name in names}
