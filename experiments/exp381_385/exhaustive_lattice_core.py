from __future__ import annotations

import math
from typing import Sequence

import numpy as np


SEVEN_SOURCES = (
    "trend",
    "dynamics",
    "spectral",
    "normal_distance",
    "spectral_derivative",
    "shapelet_normal_prototype",
    "interval_quantile",
)
NEW_TRAIN_SCORE_SOURCES = (
    "spectral_derivative",
    "shapelet_normal_prototype",
    "interval_quantile",
)


def leave_one_out_tail_candidates(
    train_scores: Sequence[float] | np.ndarray,
    alpha: float,
) -> tuple[int, ...]:
    scores = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise ValueError("TRAIN scores must be finite")
    n_train = int(scores.size)
    if n_train < 5 or math.floor(float(alpha) * n_train + 1e-12) < 1:
        return ()
    unique, counts = np.unique(scores, return_counts=True)
    count_ge_by_value: dict[float, int] = {}
    cumulative = 0
    for value, count in zip(unique[::-1], counts[::-1], strict=True):
        cumulative += int(count)
        count_ge_by_value[float(value)] = cumulative
    return tuple(
        index
        for index, score in enumerate(scores)
        if count_ge_by_value[float(score)] / n_train
        <= float(alpha) + 1e-12
    )


def wilson_upper_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z * z / (4.0 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)
