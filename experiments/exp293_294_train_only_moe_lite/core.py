from __future__ import annotations

import math
from statistics import NormalDist
from typing import Mapping

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(np.std(left_rank)) <= 1e-12 or float(np.std(right_rank)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else 0.0


def _loo_upper_pvalues(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=float).reshape(-1)
    result = np.ones(len(scores), dtype=float)
    for index, value in enumerate(scores):
        reference = np.delete(scores, index)
        result[index] = (1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0)
    return result


def family_majority_loo_false_count(train_variants: Mapping[str, np.ndarray], *, alpha: float) -> int:
    arrays = [np.asarray(values, dtype=float).reshape(-1) for _, values in sorted(train_variants.items())]
    if not arrays or len({len(values) for values in arrays}) != 1:
        raise ValueError("TRAIN variant length mismatch")
    votes = np.sum(np.vstack([_loo_upper_pvalues(values) <= float(alpha) for values in arrays]), axis=0)
    return int(np.count_nonzero(votes >= math.ceil(len(arrays) / 2)))


def _wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    p = successes / trials
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((center + radius) / denominator)


def _train_drift(values: np.ndarray) -> float:
    scores = np.asarray(values, dtype=float)
    if len(scores) < 4:
        return float("inf")
    left, right = np.array_split(scores, 2)
    scale = max(float(1.4826 * np.median(np.abs(scores - np.median(scores)))), 1e-12)
    return abs(float(np.median(left) - np.median(right))) / scale


def audit_expert(
    *, dataset: str, expert: str, train_variants: Mapping[str, np.ndarray],
    provenance_verified: bool, alpha: float,
) -> dict[str, object]:
    arrays = {name: np.asarray(values, dtype=float).reshape(-1) for name, values in train_variants.items()}
    lengths = {len(values) for values in arrays.values()}
    available = bool(arrays) and len(lengths) == 1 and next(iter(lengths), 0) >= 5
    finite = available and all(np.isfinite(values).all() for values in arrays.values())
    n_train = next(iter(lengths), 0) if len(lengths) == 1 else 0
    if not available or not finite:
        return {
            "dataset": dataset, "expert": expert, "available": False,
            "provenance_verified": bool(provenance_verified), "n_train": int(n_train),
            "variant_count": len(arrays), "normal_false_count": 0,
            "normal_false_rate": 0.0, "risk_upper_95": 1.0, "stability": 0.0,
            "attainable_p": 1.0, "train_drift": float("inf"),
            "contains_test_values": False, "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
    names = sorted(arrays)
    correlations = [abs(_spearman(arrays[left], arrays[right])) for i, left in enumerate(names) for right in names[i + 1 :]]
    stability = float(np.median(correlations)) if correlations else 1.0
    false_count = family_majority_loo_false_count(arrays, alpha=alpha)
    drifts = [_train_drift(values) for values in arrays.values()]
    return {
        "dataset": dataset, "expert": expert, "available": True,
        "provenance_verified": bool(provenance_verified), "n_train": int(n_train),
        "variant_count": len(arrays), "normal_false_count": false_count,
        "normal_false_rate": false_count / n_train,
        "risk_upper_95": _wilson_upper(false_count, n_train),
        "stability": stability, "attainable_p": 1.0 / n_train,
        "train_drift": float(np.median(drifts)),
        "contains_test_values": False, "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
