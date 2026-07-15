"""Pure TRAIN-only helpers for Exp153--157.

The module intentionally has no database access and no TEST-label access.  It
contains only deterministic statistical helpers used by the audit, pooling,
fusion, confirmation and risk-control experiments.  Keeping these functions
pure makes it possible to test the policy contract without the 1.6 GB local
SQLite database.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import numpy as np

SEED = 20260717
ALPHAS = (0.005, 0.01, 0.02, 0.05)
REQUIRED_SOURCE_AGREEMENT = 2
RISK_TARGETS_PER_RUN = (0.005, 0.01, 0.02)  # 0.5, 1 and 2 false alerts / 100 runs
RISK_CONFIDENCE = 0.95  # one-sided Wilson upper confidence bound
MIN_POOL_REFERENCE = 200  # enough reference points to resolve alpha=0.005

CONFIRMATION_VARIANTS = (
    {
        "name": "legacy_countcap",
        "block_b_method": "countcap",
        "block_b_level": 0.015,
        "block_c_method": "countcap",
        "block_c_level": 0.01,
    },
    {
        "name": "conformal_strict",
        "block_b_method": "conformal",
        "block_b_level": 0.01,
        "block_c_method": "conformal",
        "block_c_level": 0.02,
    },
    {
        "name": "conformal_balanced",
        "block_b_method": "conformal",
        "block_b_level": 0.02,
        "block_c_method": "conformal",
        "block_c_level": 0.05,
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_file(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            "frozen artifact hash mismatch: "
            f"expected={expected_sha256} actual={actual} path={path}"
        )


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def empirical_conformal_p_values(
    reference_scores: Iterable[float], query_scores: Iterable[float]
) -> np.ndarray:
    """One-sided empirical conformal p-values with the finite-sample +1 rule."""

    reference = finite(reference_scores)
    query = np.asarray(list(query_scores), dtype=float)
    if len(reference) == 0:
        return np.ones(len(query), dtype=float)
    return np.asarray(
        [
            1.0
            if not np.isfinite(score)
            else (1.0 + float(np.sum(reference >= score))) / (len(reference) + 1.0)
            for score in query
        ],
        dtype=float,
    )


def leave_one_out_conformal_p_values(scores: Iterable[float]) -> np.ndarray:
    """Pseudo-normal p-values where each TRAIN OOF score excludes itself.

    The input scores must already be out-of-fold scores, so the model that
    produced score ``i`` did not fit instance ``i``.  Removing the score from
    its own empirical reference avoids a second, obvious self-comparison.
    """

    values = np.asarray(list(scores), dtype=float)
    output = np.ones(len(values), dtype=float)
    for index, score in enumerate(values):
        reference = np.delete(values, index)
        reference = reference[np.isfinite(reference)]
        if len(reference) and np.isfinite(score):
            output[index] = (1.0 + float(np.sum(reference >= score))) / (
                len(reference) + 1.0
            )
    return output


def robust_location_scale(reference_scores: Iterable[float]) -> tuple[float, float]:
    """Return a robust TRAIN-only center and scale.

    MAD is preferred.  Constant references fall back to IQR, standard deviation
    and finally 1.0.  The fallback order is deterministic and never looks at
    TEST values.
    """

    reference = finite(reference_scores)
    if len(reference) == 0:
        return 0.0, 1.0
    location = float(np.median(reference))
    mad = float(np.median(np.abs(reference - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        q75, q25 = np.percentile(reference, [75, 25])
        scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(reference))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return location, scale


def robust_standardize(
    reference_scores: Iterable[float], values: Iterable[float]
) -> tuple[np.ndarray, float, float]:
    location, scale = robust_location_scale(reference_scores)
    array = np.asarray(list(values), dtype=float)
    return (array - location) / scale, location, scale


def leave_one_dataset_out_pool(
    standardized_train_scores: Mapping[str, np.ndarray], target_dataset: str
) -> np.ndarray:
    """Concatenate TRAIN-normal standardized scores from all other datasets."""

    pieces = [
        np.asarray(scores, dtype=float)
        for dataset, scores in sorted(standardized_train_scores.items())
        if dataset != target_dataset and len(np.asarray(scores))
    ]
    if not pieces:
        return np.asarray([], dtype=float)
    return np.concatenate(pieces)


def excluded_pool_conformal_p_values(
    global_standardized_reference: Iterable[float],
    target_standardized_reference: Iterable[float],
    query_standardized_scores: Iterable[float],
) -> np.ndarray:
    """Conformal p-values against a leave-one-dataset-out pooled reference.

    ``global_standardized_reference`` contains every dataset. The target
    dataset's TRAIN-normal standardized scores are subtracted from the tail
    counts and denominator without materializing a full copied pool.
    """

    global_ref = finite(global_standardized_reference)
    target_ref = finite(target_standardized_reference)
    query = np.asarray(list(query_standardized_scores), dtype=float)
    remaining = len(global_ref) - len(target_ref)
    if remaining <= 0:
        return np.ones(len(query), dtype=float)
    output = []
    for score in query:
        if not np.isfinite(score):
            output.append(1.0)
            continue
        global_tail = int(np.sum(global_ref >= score))
        target_tail = int(np.sum(target_ref >= score))
        tail = max(0, global_tail - target_tail)
        output.append((1.0 + tail) / (remaining + 1.0))
    return np.asarray(output, dtype=float)


def local_minimum_p(reference_scores: Iterable[float]) -> float:
    reference = finite(reference_scores)
    return 1.0 / (len(reference) + 1.0) if len(reference) else 1.0


def hybrid_conformal_p_values(
    local_reference: Iterable[float],
    query_scores: Iterable[float],
    pooled_standardized_reference: Iterable[float],
    *,
    alpha: float,
    minimum_pool_size: int = MIN_POOL_REFERENCE,
) -> tuple[np.ndarray, str]:
    """Use local conformal when resolvable, otherwise a robust pooled fallback.

    Pooled values are assumed to have been robust-standardized within their own
    datasets.  The target query is standardized only with its target TRAIN
    reference before comparison to the leave-one-dataset-out pool.
    """

    local = finite(local_reference)
    query = np.asarray(list(query_scores), dtype=float)
    if len(local) and local_minimum_p(local) <= float(alpha):
        return empirical_conformal_p_values(local, query), "local"

    pool = finite(pooled_standardized_reference)
    if len(pool) < int(minimum_pool_size) or len(local) == 0:
        return np.ones(len(query), dtype=float), "unavailable"

    query_z, _, _ = robust_standardize(local, query)
    return empirical_conformal_p_values(pool, query_z), "pooled"


def train_count_bin(n_train: int) -> str:
    value = int(n_train)
    if value < 5:
        return "n000_004"
    if value < 20:
        return "n005_019"
    if value < 50:
        return "n020_049"
    if value < 100:
        return "n050_099"
    return "n100_plus"


def length_quantile_bin(train_target_len: int, boundaries: Sequence[float]) -> str:
    value = float(train_target_len)
    q1, q2, q3 = [float(item) for item in boundaries]
    if value <= q1:
        return "len_q1"
    if value <= q2:
        return "len_q2"
    if value <= q3:
        return "len_q3"
    return "len_q4"


def train_only_group_key(
    n_train: int, train_target_len: int, length_boundaries: Sequence[float]
) -> str:
    """Group datasets using TRAIN-only sample count and TRAIN length."""

    return f"{train_count_bin(n_train)}|{length_quantile_bin(train_target_len, length_boundaries)}"


def agreement_indices(
    p_values_by_source: Mapping[str, np.ndarray],
    alpha: float,
    required_sources: int = REQUIRED_SOURCE_AGREEMENT,
) -> tuple[set[int], np.ndarray]:
    """Select indices supported by the required number of score sources."""

    names = sorted(p_values_by_source)
    if not names:
        return set(), np.zeros(0, dtype=int)
    lengths = {len(np.asarray(p_values_by_source[name])) for name in names}
    if len(lengths) != 1:
        raise ValueError(f"source p-value length mismatch: {sorted(lengths)}")
    matrix = np.vstack(
        [np.asarray(p_values_by_source[name], dtype=float) <= float(alpha) for name in names]
    )
    counts = matrix.sum(axis=0)
    selected = set(
        np.flatnonzero(counts >= int(required_sources)).astype(int).tolist()
    )
    return selected, counts


def rankdata_average(values: Iterable[float]) -> np.ndarray:
    """Average ranks for ties, implemented without scipy."""

    array = np.asarray(list(values), dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_correlation_matrix(score_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("score_matrix must be two-dimensional")
    if matrix.shape[1] == 0:
        return np.zeros((0, 0), dtype=float)
    ranked = np.column_stack([rankdata_average(matrix[:, i]) for i in range(matrix.shape[1])])
    if matrix.shape[0] < 2:
        return np.eye(matrix.shape[1], dtype=float)
    corr = np.corrcoef(ranked, rowvar=False)
    corr = np.asarray(corr, dtype=float)
    if corr.ndim == 0:
        corr = np.asarray([[1.0]], dtype=float)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def inverse_correlation_weights(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TRAIN-only weights that downweight sources redundant with the others."""

    corr = spearman_correlation_matrix(score_matrix)
    source_count = corr.shape[0]
    if source_count == 0:
        return np.asarray([], dtype=float), corr
    if source_count == 1:
        return np.ones(1, dtype=float), corr
    redundancy = (np.sum(np.abs(corr), axis=1) - 1.0) / (source_count - 1)
    raw = 1.0 / (1.0 + redundancy)
    weights = raw / np.mean(raw)
    return weights, corr


def top2_mean_score(score_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("top2_mean_score requires at least two sources")
    sorted_values = np.sort(matrix, axis=1)
    return np.mean(sorted_values[:, -2:], axis=1)


def weighted_top2_mean_score(score_matrix: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    weight_array = np.asarray(list(weights), dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(weight_array):
        raise ValueError("weights must match score_matrix columns")
    return top2_mean_score(matrix * weight_array.reshape(1, -1))


def meta_conformal_p_values(
    train_score_matrix: np.ndarray,
    query_score_matrix: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate a continuous multi-source meta-score using TRAIN only."""

    train_matrix = np.asarray(train_score_matrix, dtype=float)
    query_matrix = np.asarray(query_score_matrix, dtype=float)
    if train_matrix.ndim != 2 or query_matrix.ndim != 2:
        raise ValueError("train/query score matrices must be two-dimensional")
    if train_matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError("train/query source counts differ")

    if method == "top2_mean":
        meta_train = top2_mean_score(train_matrix)
        meta_query = top2_mean_score(query_matrix)
    elif method == "corr_weighted_top2":
        weights, _ = inverse_correlation_weights(train_matrix)
        meta_train = weighted_top2_mean_score(train_matrix, weights)
        meta_query = weighted_top2_mean_score(query_matrix, weights)
    else:
        raise ValueError(f"unknown fusion method: {method}")

    return empirical_conformal_p_values(meta_train, meta_query), meta_train, meta_query


def countcap_threshold(reference_scores: Iterable[float], rate: float) -> float:
    """Small pure equivalent of the historical quantile + count-cap threshold."""

    reference = finite(reference_scores)
    if len(reference) == 0:
        return math.inf
    rate = min(max(float(rate), 0.0), 1.0)
    quantile_threshold = float(np.quantile(reference, 1.0 - rate))
    cap = max(1, int(math.ceil(len(reference) * rate)))
    ordered = np.sort(reference)
    cap_index = max(0, len(ordered) - cap - 1)
    cap_threshold = float(ordered[cap_index])
    return max(quantile_threshold, cap_threshold)


def leave_one_out_countcap_flags(scores: Iterable[float], rate: float) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    flags = np.zeros(len(values), dtype=bool)
    for index, score in enumerate(values):
        threshold = countcap_threshold(np.delete(values, index), rate)
        flags[index] = bool(np.isfinite(score) and score > threshold)
    return flags


def route_lanes(candidates, block_b, block_c, test_size: int) -> dict[str, set[int]]:
    candidates = {int(i) for i in candidates}
    block_b = {int(i) for i in block_b}
    block_c = {int(i) for i in block_c}
    universe = set(range(int(test_size)))
    invalid = sorted((candidates | block_b | block_c) - universe)
    if invalid:
        raise ValueError(f"out-of-bounds indices: {invalid[:10]}")
    hard = candidates & block_b
    remaining = candidates - hard
    priority = remaining & block_c
    standard = remaining - priority
    return {
        "hard": hard,
        "standard_review": standard,
        "priority_review": priority,
        "no_alert": universe - candidates,
    }


def wilson_upper_bound(
    successes: int,
    trials: int,
    *,
    confidence: float = RISK_CONFIDENCE,
) -> float:
    """One-sided Wilson upper confidence bound for a Bernoulli rate."""

    successes = int(successes)
    trials = int(trials)
    if trials <= 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    z = NormalDist().inv_cdf(float(confidence))
    p_hat = successes / trials
    denominator = 1.0 + (z * z) / trials
    center = p_hat + (z * z) / (2.0 * trials)
    radius = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / trials) + (z * z) / (4.0 * trials * trials)
    )
    return min(1.0, (center + radius) / denominator)


def select_largest_alpha_by_risk(
    audit_rows: Sequence[Mapping[str, float]],
    *,
    target_rate: float,
    confidence: float = RISK_CONFIDENCE,
) -> dict[str, object]:
    """Select the largest pre-registered alpha meeting a TRAIN-only risk bound."""

    evaluated = []
    for row in sorted(audit_rows, key=lambda item: float(item["alpha"])):
        trials = int(row["pseudo_runs"])
        false_alerts = int(row["hard_false_alerts"])
        upper = wilson_upper_bound(false_alerts, trials, confidence=confidence)
        evaluated.append(
            {
                "alpha": float(row["alpha"]),
                "pseudo_runs": trials,
                "hard_false_alerts": false_alerts,
                "observed_rate": false_alerts / trials if trials else 0.0,
                "upper_bound": upper,
                "eligible": bool(trials > 0 and upper <= float(target_rate)),
            }
        )
    eligible = [item for item in evaluated if item["eligible"]]
    selected_alpha = max((item["alpha"] for item in eligible), default=None)
    return {
        "target_rate": float(target_rate),
        "confidence": float(confidence),
        "selected_alpha": selected_alpha,
        "abstain": selected_alpha is None,
        "evaluated": evaluated,
    }


def format_indices(indices: Iterable[int]) -> str:
    return " ".join(str(int(index)) for index in sorted(set(indices)))


def parse_indices(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    text = str(value).replace(",", " ").strip()
    return {int(token) for token in text.split() if token}


def tier_metrics(y_test: Iterable[int], indices: Iterable[int]) -> dict[str, float]:
    labels = np.asarray(list(y_test), dtype=int)
    selected = sorted(set(int(i) for i in indices))
    if any(index < 0 or index >= len(labels) for index in selected):
        raise ValueError("metric index out of bounds")
    tp = int(np.sum(labels[selected])) if selected else 0
    fp = len(selected) - tp
    total_positive = int(np.sum(labels))
    fn = total_positive - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
