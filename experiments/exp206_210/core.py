import hashlib
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping

import numpy as np

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ALPHAS = (0.005, 0.01, 0.02, 0.05)
SHADOW_RISK = 0.001
REVIEW_RISK = 0.005
SEED = 20260716
SOURCE_NAMES = ("spectral", "shapelet", "interval")

FORBIDDEN_PREDICTION_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions", "zero_f1",
    "tp", "fp", "fn", "precision", "recall", "f1", "hard_tp",
    "hard_fp", "hard_fn", "hard_f1", "combined_f1",
}


def fold_plan(n: int, folds: int = 5, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    split_indices = np.array_split(indices, folds)
    
    fold_list = []
    for i in range(folds):
        query_idx = split_indices[i]
        fit_parts = [split_indices[j] for j in range(folds) if j != i]
        fit_idx = np.concatenate(fit_parts)
        fold_list.append((np.sort(fit_idx), np.sort(query_idx)))
    return fold_list


def empirical_upper_pvalues(reference: Iterable[float], query: Iterable[float]) -> np.ndarray:
    ref = np.asarray(list(reference), dtype=float)
    values = np.asarray(list(query), dtype=float)
    ref = ref[np.isfinite(ref)]
    if not len(ref):
        return np.ones(len(values), dtype=float)
    ordered = np.sort(ref)
    counts = len(ordered) - np.searchsorted(ordered, values, side="left")
    result = (1.0 + counts.astype(float)) / (len(ordered) + 1.0)
    result[~np.isfinite(values)] = 1.0
    return result


def leave_one_out_pvalues(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    result = np.ones(len(values), dtype=float)
    for index, value in enumerate(values):
        result[index] = empirical_upper_pvalues(
            np.delete(values, index), np.asarray([value])
        )[0]
    return result


def robust_location_scale(reference: Iterable[float]) -> tuple[float, float]:
    ref = np.asarray(list(reference), dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return 0.0, 1.0
    location = float(np.median(ref))
    mad = float(np.median(np.abs(ref - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        q75, q25 = np.percentile(ref, [75, 25])
        scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(ref))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return location, scale


def robust_z_pair(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    query = np.asarray(query, dtype=float)
    location, scale = robust_location_scale(train)
    return (train - location) / scale, (query - location) / scale


def top2_mean(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("top2_mean requires a 2D array with at least two columns")
    sorted_matrix = np.sort(matrix, axis=1)
    return sorted_matrix[:, -2:].mean(axis=1)


def rankdata_average(values: Iterable[float]) -> np.ndarray:
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


def inverse_spearman_weights(matrix: np.ndarray) -> np.ndarray:
    corr = spearman_correlation_matrix(matrix)
    source_count = corr.shape[0]
    if source_count == 0:
        return np.asarray([], dtype=float)
    if source_count == 1:
        return np.ones(1, dtype=float)
    redundancy = (np.sum(np.abs(corr), axis=1) - 1.0) / (source_count - 1)
    raw = 1.0 / (1.0 + redundancy)
    weights = raw / np.mean(raw)
    return weights


def fisher_stat(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    tiny = 1e-15
    p_clipped = np.maximum(pvalues, tiny)
    if pvalues.ndim == 2:
        return -2.0 * np.log(p_clipped).sum(axis=1)
    else:
        return -2.0 * np.log(p_clipped).sum()


def wilson_upper(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return 1.0
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    p = successes / trials
    z = NormalDist().inv_cdf(confidence)
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    return float((center + radius) / denominator)


def classify_risk(upper: float) -> str:
    if upper <= SHADOW_RISK:
        return "shadow_hard_diagnostic"
    if upper <= REVIEW_RISK:
        return "standard_review_diagnostic"
    return "research_sensitivity_only"


def validate_prediction_payload(payload: Mapping[str, object]) -> None:
    forbidden = FORBIDDEN_PREDICTION_FIELDS & {str(key).lower() for key in payload}
    if forbidden:
        raise ValueError(f"evaluation fields in frozen prediction: {sorted(forbidden)}")
    if payload.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if payload.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    n_test = int(payload["n_test"])
    indices = [int(value) for value in payload.get("candidate_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate indices")
    if indices and (min(indices) < 0 or max(indices) >= n_test):
        raise ValueError("candidate index outside TEST range")


def parse_indices(value: object) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return {int(item) for item in value}
    text = str(value).replace(",", " ").strip()
    return {int(token) for token in text.split() if token}

