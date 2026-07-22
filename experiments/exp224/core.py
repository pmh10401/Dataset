from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from experiments.exp177_185.core import _fold_plan
from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp220_223.evaluation import wilson_bound

ALPHAS = (0.005, 0.01, 0.02, 0.05)
SEED = 20260718
RISK_TARGET = 0.005


def _score_pair(
    temporal_scores: Iterable[float],
    canonical_scores: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    temporal = np.asarray(list(temporal_scores), dtype=float)
    canonical = np.asarray(list(canonical_scores), dtype=float)
    if temporal.ndim != 1 or canonical.ndim != 1:
        raise ValueError("source scores must be one-dimensional")
    if len(temporal) != len(canonical):
        raise ValueError("source score lengths must match")
    if not np.all(np.isfinite(temporal)) or not np.all(np.isfinite(canonical)):
        raise ValueError("source scores must be finite")
    return temporal, canonical


def train_fold_candidate_counts(
    temporal_oof: Iterable[float],
    canonical_oof: Iterable[float],
    alphas: Iterable[float] = ALPHAS,
    seed: int = SEED,
) -> tuple[dict[float, int], int]:
    """Count dual-source TRAIN alerts using fold-local reference scores only."""
    temporal, canonical = _score_pair(temporal_oof, canonical_oof)
    alpha_values = tuple(float(alpha) for alpha in alphas)
    if any(alpha <= 0.0 or alpha >= 1.0 for alpha in alpha_values):
        raise ValueError("alphas must be between zero and one")
    counts = {alpha: 0 for alpha in alpha_values}
    for fit_indices, query_indices in _fold_plan(len(temporal), int(seed)):
        temporal_p = empirical_upper_tail_pvalues(
            temporal[fit_indices], temporal[query_indices]
        )
        canonical_p = empirical_upper_tail_pvalues(
            canonical[fit_indices], canonical[query_indices]
        )
        for alpha in alpha_values:
            counts[alpha] += int(
                np.count_nonzero((temporal_p <= alpha) & (canonical_p <= alpha))
            )
    return counts, len(temporal)


def select_largest_safe_alpha(
    false_counts: Mapping[float, int],
    trials: int,
    target: float = RISK_TARGET,
) -> float | None:
    """Select the largest alpha whose one-sided 95% Wilson upper is safe."""
    if int(trials) <= 0:
        raise ValueError("trials must be positive")
    safe = [
        float(alpha)
        for alpha, count in false_counts.items()
        if wilson_bound(int(count), int(trials), side="upper") <= float(target)
    ]
    return max(safe) if safe else None


def route_exp224(
    temporal_p: Iterable[float],
    canonical_p: Iterable[float],
    alpha: float | None,
    exp137_excluded: Iterable[int],
) -> set[int]:
    """Return dual-source candidates after excluding every Exp137 lane."""
    if alpha is None:
        return set()
    temporal, canonical = _score_pair(temporal_p, canonical_p)
    selected = set(
        np.flatnonzero((temporal <= float(alpha)) & (canonical <= float(alpha))).tolist()
    )
    return selected - {int(index) for index in exp137_excluded}
