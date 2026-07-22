from __future__ import annotations

from collections.abc import Mapping

import numpy as np


ALPHAS = (0.005, 0.01, 0.02, 0.05)
RANK_DEPTHS = (1, 2, 3, 5, 8, 10)
SOURCE_NAMES = (
    "temporal_random_kernel",
    "spectral_image",
    "texture_image",
    "interval_statistical",
)
SOURCE_CACHE_KEYS = {
    "temporal_random_kernel": "rocket_exp40",
    "spectral_image": "exp55_best",
    "texture_image": "exp56_best",
    "interval_statistical": "exp84",
}
CAUSE_CATEGORIES = (
    "already_in_standard_review",
    "already_in_priority_review",
    "score_visible_policy_miss",
    "single_source_only",
    "representation_blind",
)
FORBIDDEN_FROZEN_FIELDS = {
    "y_test",
    "labels",
    "test_labels",
    "tp",
    "fp",
    "fn",
    "f1",
    "hard_tp",
    "hard_fp",
    "hard_fn",
    "hard_f1",
    "anomaly_count",
}


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    """Return one-based descending ranks with stable index-order tie breaking."""

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional array")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int64)
    return ranks


def upper_tail_pvalue(train_scores: np.ndarray, value: float) -> float:
    """Finite-sample upper-tail p-value against a frozen TRAIN OOF reference."""

    reference = np.asarray(train_scores, dtype=float)
    if reference.ndim != 1 or not len(reference) or not np.all(np.isfinite(reference)):
        raise ValueError("TRAIN reference must be a non-empty finite vector")
    if not np.isfinite(value):
        raise ValueError("query score must be finite")
    return float((1 + np.count_nonzero(reference >= float(value))) / (len(reference) + 1))


def source_diagnostic(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    index: int,
) -> dict[str, float | int]:
    test = np.asarray(test_scores, dtype=float)
    if index < 0 or index >= len(test):
        raise IndexError(index)
    ranks = descending_ranks(test)
    pvalue = upper_tail_pvalue(train_scores, float(test[index]))
    result: dict[str, float | int] = {
        "score": float(test[index]),
        "rank": int(ranks[index]),
        "rank_percentile": float(ranks[index] / max(1, len(test))),
        "pvalue": pvalue,
    }
    for alpha in ALPHAS:
        result[f"support_alpha_{alpha:g}"] = int(pvalue <= alpha)
    return result


def classify_miss(
    in_standard_review: bool,
    in_priority_review: bool,
    support_counts: Mapping[float, int],
) -> str:
    if in_standard_review:
        return "already_in_standard_review"
    if in_priority_review:
        return "already_in_priority_review"
    support = int(support_counts.get(0.05, 0))
    if support >= 2:
        return "score_visible_policy_miss"
    if support == 1:
        return "single_source_only"
    return "representation_blind"


def validate_frozen_record(record: Mapping[str, object]) -> None:
    forbidden = FORBIDDEN_FROZEN_FIELDS & {str(key).lower() for key in record}
    if forbidden:
        raise ValueError(f"evaluation field in frozen record: {sorted(forbidden)}")
    n_train = int(record["n_train"])
    n_test = int(record["n_test"])
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_NAMES):
        raise ValueError("frozen record must contain all four score sources")
    for source_name in SOURCE_NAMES:
        source = sources[source_name]
        if not isinstance(source, Mapping):
            raise ValueError(f"invalid source record: {source_name}")
        train = np.asarray(source["train"], dtype=float)
        test = np.asarray(source["test"], dtype=float)
        if len(train) != n_train:
            raise ValueError(f"TRAIN score length mismatch: {source_name}")
        if len(test) != n_test:
            raise ValueError(f"TEST score length mismatch: {source_name}")
        if not np.all(np.isfinite(train)) or not np.all(np.isfinite(test)):
            raise ValueError(f"non-finite score: {source_name}")


def mean_rank(test_scores_by_source: Mapping[str, np.ndarray]) -> np.ndarray:
    ranks = [descending_ranks(test_scores_by_source[name]) for name in SOURCE_NAMES]
    return np.mean(np.vstack(ranks), axis=0)


def counterfactual_metrics(
    labels: np.ndarray,
    candidate_indices: set[int],
    hard_indices: set[int],
) -> dict[str, int | float]:
    truth = set(np.flatnonzero(np.asarray(labels) != 0).astype(int).tolist())
    candidates = {int(index) for index in candidate_indices}
    hard = {int(index) for index in hard_indices}
    if candidates and (min(candidates) < 0 or max(candidates) >= len(labels)):
        raise ValueError("candidate index outside TEST range")
    tp = len(candidates & truth)
    fp = len(candidates - truth)
    return {
        "candidate_count": len(candidates),
        "tp": tp,
        "fp": fp,
        "precision": tp / max(1, tp + fp),
        "hard_miss_recovered": len(candidates & (truth - hard)),
    }


def multiplicity_bin(anomaly_count: int) -> str:
    count = int(anomaly_count)
    if count <= 1:
        return "1"
    if count <= 3:
        return "2_3"
    if count <= 10:
        return "4_10"
    return "11_plus"


def test_size_bin(test_size: int) -> str:
    size = int(test_size)
    if size < 100:
        return "lt100"
    if size < 250:
        return "100_249"
    if size < 500:
        return "250_499"
    return "500_plus"
