"""Pure NumPy reference scoring for Exp417 persistent horizon residuals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real

import numpy as np

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import HorizonScore, SeedDecision


_MAD_TO_NORMAL_SCALE = 1.4826
_MIN_ROBUST_SCALE = 1e-12


def longest_true_run_fraction(mask: np.ndarray) -> float:
    """Return the longest contiguous true run divided by the exact mask length."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.dtype(bool):
        raise ValueError("mask must be a one-dimensional boolean ndarray")
    if mask.size == 0:
        return 0.0
    longest = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest / mask.size)


def empirical_p_value(oof_scores: Sequence[float], query: float) -> float:
    """Compute the frozen upper-tail empirical p-value with plus-one correction."""
    values = _finite_score_sequence(oof_scores, "oof_scores")
    query_value = _finite_number(query, "query")
    return float((1 + np.count_nonzero(values >= query_value)) / (1 + values.size))


def score_horizon(
    oof_residuals: Sequence[np.ndarray], query_residual: np.ndarray, *, horizon: int
) -> HorizonScore:
    """Score one exact, aligned horizon against only its normal OOF residuals."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    oof = _aligned_residual_series(oof_residuals, "oof_residuals")
    query = _residual_vector(query_residual, "query_residual")
    length = oof[0].size
    if query.size != length or any(values.size != length for values in oof):
        raise ValueError("residuals must preserve the exact aligned length")

    pooled = np.concatenate(oof)
    center = float(np.median(pooled))
    scale = float(_MAD_TO_NORMAL_SCALE * np.median(np.abs(pooled - center)))
    if not isfinite(center) or not isfinite(scale) or scale <= _MIN_ROBUST_SCALE:
        raise ValueError("OOF residual robust scale must be finite and greater than 1e-12")

    normalized_oof = tuple((values - center) / scale for values in oof)
    normalized_query = (query - center) / scale
    if not all(np.isfinite(values).all() for values in (*normalized_oof, normalized_query)):
        raise ValueError("standardized residuals must be finite")

    exceedance_threshold = float(np.percentile(np.concatenate(normalized_oof), 99, method="linear"))
    oof_scores = tuple(
        _series_score(values, exceedance_threshold)[0] for values in normalized_oof
    )
    query_score, query_persistence = _series_score(normalized_query, exceedance_threshold)
    return HorizonScore(
        horizon=horizon,
        score=query_score,
        p_value=empirical_p_value(oof_scores, query_score),
        persistence=query_persistence,
    )


def seed_decision(
    scores: Mapping[int, HorizonScore], *, seed: int, config: Exp417FrozenConfig = FROZEN_CONFIG
) -> SeedDecision:
    """Construct one frozen-seed decision from its three immutable horizon scores."""
    if type(seed) is not int or seed < 1:
        raise ValueError("seed must be a positive integer")
    if not isinstance(config, Exp417FrozenConfig):
        raise ValueError("config must be an Exp417 frozen config")
    config.validate()
    if seed not in config.model_seeds:
        raise ValueError("seed must be one of the exact frozen model seeds")
    frozen_scores = _validated_horizon_scores(scores)
    candidate = sum(
        item.p_value <= config.candidate_alpha
        and item.persistence >= config.persistence_fraction
        for item in frozen_scores.values()
    ) >= 2
    return SeedDecision(seed=seed, scores=frozen_scores, candidate=candidate, config=config)


def consensus_candidate(decisions: Sequence[SeedDecision]) -> bool:
    """Require precisely the frozen seeds and a positive decision from each one."""
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise ValueError("decisions must be a sequence")
    frozen = tuple(decisions)
    if len(frozen) != len(FROZEN_CONFIG.model_seeds) or any(
        not isinstance(item, SeedDecision) for item in frozen
    ):
        raise ValueError("decisions must contain exactly three SeedDecision records")
    seeds = tuple(item.seed for item in frozen)
    if len(set(seeds)) != len(seeds) or set(seeds) != set(FROZEN_CONFIG.model_seeds):
        raise ValueError("decisions must contain the three exact frozen model seeds")
    horizons = tuple(frozen[0].scores)
    if any(tuple(item.scores) != horizons for item in frozen[1:]):
        raise ValueError("decisions must contain identical exact horizon evidence")
    return all(item.candidate for item in frozen)


def _finite_score_sequence(values: Sequence[float], name: str) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    copied = tuple(values)
    if not copied:
        raise ValueError(f"{name} must be non-empty")
    return np.asarray([_finite_number(value, name) for value in copied], dtype=np.float64)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must contain finite real numbers")
    return float(value)


def _aligned_residual_series(values: Sequence[np.ndarray], name: str) -> tuple[np.ndarray, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence")
    copied = tuple(values)
    if not copied:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_residual_vector(item, name) for item in copied)


def _residual_vector(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional ndarray")
    if values.dtype == np.dtype(bool) or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must contain numeric residuals")
    if np.issubdtype(values.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real residuals")
    copied = np.asarray(values, dtype=np.float64)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} must be finite")
    if np.any(copied < 0.0):
        raise ValueError(f"{name} must contain absolute residuals")
    return copied


def _series_score(values: np.ndarray, threshold: float) -> tuple[float, float]:
    q95 = float(np.percentile(values, 95, method="linear"))
    persistence = longest_true_run_fraction(values >= threshold)
    score = float(q95 + (2.0 * persistence))
    if not isfinite(score):
        raise ValueError("series score must be finite")
    return score, persistence


def _validated_horizon_scores(scores: Mapping[int, HorizonScore]) -> Mapping[int, HorizonScore]:
    if not isinstance(scores, Mapping):
        raise ValueError("scores must be a mapping")
    if any(type(horizon) is not int or horizon < 1 for horizon in scores):
        raise ValueError("scores must use positive integer horizons")
    ordered = tuple(sorted(scores))
    if len(ordered) != 3 or len(set(ordered)) != 3:
        raise ValueError("scores must contain exactly three horizons")
    if not _is_possible_frozen_horizon_triple(ordered):
        raise ValueError("scores must contain one possible exact frozen horizon triple")
    frozen = {horizon: scores[horizon] for horizon in ordered}
    if any(not isinstance(item, HorizonScore) or item.horizon != horizon for horizon, item in frozen.items()):
        raise ValueError("scores must map exact horizon keys to matching HorizonScore evidence")
    return frozen


def _is_possible_frozen_horizon_triple(horizons: tuple[int, int, int]) -> bool:
    """Check that the IDs can arise from one unpadded length group.

    The score boundary has no series length, so it cannot prove the exact
    group's length.  It does reject IDs that cannot be generated by the frozen
    horizon formula; ``ShadowCandidate`` binds the retained evidence to its
    exact length later in the workflow.
    """
    short, medium, long = horizons
    if short != 1 or len({short, medium, long}) != 3:
        return False
    lower = max((50 * (medium - 1)) + 1, (10 * (long - 1)) + 1)
    upper = min(50 * medium, 10 * long)
    return lower <= upper


__all__ = [
    "consensus_candidate",
    "empirical_p_value",
    "longest_true_run_fraction",
    "score_horizon",
    "seed_decision",
]
