from __future__ import annotations
from statistics import NormalDist

import numpy as np

def n_train_bin(n_train: int) -> str:
    if n_train < 5:
        return "lt5"
    if n_train < 20:
        return "5_19"
    if n_train < 50:
        return "20_49"
    if n_train < 100:
        return "50_99"
    return "ge100"

def fold_plan(n_train: int, seed: int):
    """
    Returns fold indices and splits for cross-fitting.
    n_train < 5: returns empty list (abstention)
    5 <= n_train < 20: leave-one-out
    n_train >= 20: shuffled 5-fold (seed 20260715)
    """
    if n_train < 5:
        return []

    indices = np.arange(n_train)
    if n_train < 20:
        # deterministic leave-one-out
        plan = []
        for i in range(n_train):
            fit_idx = np.delete(indices, i)
            query_idx = np.array([i])
            plan.append((fit_idx, query_idx))
        return plan
    else:
        # shuffled deterministic 5-fold
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        # split into 5 parts
        folds = np.array_split(shuffled, 5)
        plan = []
        for i in range(5):
            query_idx = folds[i]
            # fit_idx is the concatenation of the other folds
            fit_parts = [folds[j] for j in range(5) if j != i]
            fit_idx = np.concatenate(fit_parts)
            plan.append((fit_idx, query_idx))
        return plan

def higher_threshold(fit_scores: np.ndarray, alpha: float) -> float:
    """
    Returns threshold = np.quantile(fit_scores, 1 - alpha, method="higher")
    """
    finite_scores = np.asarray(fit_scores, dtype=np.float64)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if finite_scores.size == 0:
        raise ValueError("No finite fit scores are available for calibration")
    return float(np.quantile(finite_scores, 1.0 - alpha, method="higher"))

def source_candidate_flags(query_scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns boolean array of query_scores > threshold.
    Strictly query_score > threshold.
    """
    query_scores = np.asarray(query_scores, dtype=np.float64)
    return np.isfinite(query_scores) & (query_scores > threshold)

def agreement_flags(source_flags: list[np.ndarray], required_sources: int) -> np.ndarray:
    """
    Returns agreement flag: sum(source_flags) >= required_sources
    """
    stacked = np.stack(source_flags, axis=0)
    return stacked.sum(axis=0) >= required_sources

def wilson_upper_bound(false_reviews: int, runs: int, confidence: float = 0.95) -> float:
    """
    Computes the one-sided confidence Wilson upper bound for false-review rate.
    Uses z = 1.6448536269514722 for 95% one-sided.
    """
    if runs < 0 or false_reviews < 0 or false_reviews > runs:
        raise ValueError("false_reviews and runs must satisfy 0 <= false_reviews <= runs")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")
    if runs == 0:
        return 1.0
    p = false_reviews / runs
    z = NormalDist().inv_cdf(confidence)
    denom = 1.0 + (z**2) / runs
    center = p + (z**2) / (2.0 * runs)
    spread = z * np.sqrt((p * (1.0 - p)) / runs + (z**2) / (4.0 * (runs**2)))
    upper = (center + spread) / denom
    return float(upper)

def risk_sensitivity_rows(false_reviews: int, runs: int) -> dict[float, dict[str, float | bool]]:
    """
    Evaluates sensitivity for risk targets (0.005, 0.01, 0.02).
    """
    upper_bound = wilson_upper_bound(false_reviews, runs, 0.95)
    targets = [0.005, 0.01, 0.02]
    res = {}
    for target in targets:
        res[target] = {
            "false_review_rate": false_reviews / runs if runs > 0 else 0.0,
            "wilson_upper_bound": upper_bound,
            "meets_target": upper_bound <= target
        }
    return res
