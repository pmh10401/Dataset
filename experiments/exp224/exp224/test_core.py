from __future__ import annotations

import numpy as np

from experiments.exp177_185.core import _fold_plan
from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp224.core import (
    ALPHAS,
    SEED,
    route_exp224,
    select_largest_safe_alpha,
    train_fold_candidate_counts,
)


def test_train_counts_calibrate_each_query_against_its_fit_fold_only() -> None:
    temporal = np.arange(100, dtype=float)
    canonical = np.arange(100, dtype=float)

    actual, trials = train_fold_candidate_counts(temporal, canonical)

    expected = {alpha: 0 for alpha in ALPHAS}
    for fit_indices, query_indices in _fold_plan(100, SEED):
        temporal_p = empirical_upper_tail_pvalues(
            temporal[fit_indices], temporal[query_indices]
        )
        canonical_p = empirical_upper_tail_pvalues(
            canonical[fit_indices], canonical[query_indices]
        )
        for alpha in ALPHAS:
            expected[alpha] += int(
                np.count_nonzero((temporal_p <= alpha) & (canonical_p <= alpha))
            )

    assert trials == 100
    assert actual == expected


def test_selects_largest_alpha_below_train_wilson_target() -> None:
    counts = {0.005: 0, 0.01: 0, 0.02: 1, 0.05: 60}

    assert select_largest_safe_alpha(counts, trials=10_000) == 0.02


def test_no_safe_alpha_abstains() -> None:
    counts = {alpha: 50 for alpha in ALPHAS}

    assert select_largest_safe_alpha(counts, trials=100) is None


def test_route_requires_both_sources_and_excludes_exp137() -> None:
    selected = route_exp224(
        np.array([0.001, 0.001, 0.5]),
        np.array([0.001, 0.5, 0.001]),
        0.005,
        {0},
    )

    assert selected == set()


def test_route_returns_only_dual_source_candidates() -> None:
    selected = route_exp224(
        np.array([0.001, 0.001, 0.5]),
        np.array([0.001, 0.5, 0.001]),
        0.005,
        set(),
    )

    assert selected == {0}
