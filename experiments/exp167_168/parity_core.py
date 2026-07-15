from __future__ import annotations

import numpy as np


ALPHAS = (0.005, 0.01, 0.02, 0.05)
SOURCES = ("spectral", "shapelet", "interval")
SOURCE_PAIRS = {
    "spectral+shapelet": ("spectral", "shapelet"),
    "spectral+interval": ("spectral", "interval"),
    "shapelet+interval": ("shapelet", "interval"),
}
SOURCE_CONFIGS = (*SOURCE_PAIRS, "2of3", "3of3")


def fold_paired_p_values(
    heldout_scores: np.ndarray,
    test_scores: np.ndarray,
) -> np.ndarray:
    heldout = np.asarray(heldout_scores, dtype=np.float64)
    test = np.asarray(test_scores, dtype=np.float64)
    if heldout.ndim != 1 or test.ndim != 2 or test.shape[0] != heldout.size:
        raise ValueError("fold score shape mismatch")
    if heldout.size == 0 or not np.isfinite(heldout).all() or not np.isfinite(test).all():
        raise ValueError("fold scores must be finite and non-empty")
    exceedance = (heldout[:, None] >= test).sum(axis=0)
    return (1.0 + exceedance) / (heldout.size + 1.0)


def resolution_blocked(n_train: int, alpha: float) -> bool:
    if n_train < 1:
        raise ValueError("n_train must be positive")
    return float(alpha) < 1.0 / (int(n_train) + 1.0)


def candidate_rows(
    *,
    dataset: str,
    n_train: int,
    n_test: int,
    p_values: dict[str, np.ndarray],
    config_hash: str,
) -> list[dict]:
    source_p = {
        source: np.asarray(p_values[source], dtype=np.float64) for source in SOURCES
    }
    if any(values.shape != (n_test,) for values in source_p.values()):
        raise ValueError("source p-value shape mismatch")
    if any(not np.isfinite(values).all() for values in source_p.values()):
        raise ValueError("source p-values must be finite")

    rows: list[dict] = []
    minimum_p_value = 1.0 / (int(n_train) + 1.0)
    for alpha in ALPHAS:
        flags = {source: source_p[source] <= alpha for source in SOURCES}
        source_count = np.vstack([flags[source] for source in SOURCES]).sum(axis=0)
        config_flags = {
            name: flags[sources[0]] & flags[sources[1]]
            for name, sources in SOURCE_PAIRS.items()
        }
        config_flags["2of3"] = source_count >= 2
        config_flags["3of3"] = source_count >= 3
        for source_config in SOURCE_CONFIGS:
            indices = np.flatnonzero(config_flags[source_config]).astype(int).tolist()
            row = {
                "dataset": dataset,
                "n_train": int(n_train),
                "n_test": int(n_test),
                "alpha": float(alpha),
                "source_config": source_config,
                "candidate_indices": indices,
                "candidate_count": len(indices),
                "source_candidate_counts": {
                    source: int(flags[source].sum()) for source in SOURCES
                },
                "minimum_p_value": minimum_p_value,
                "resolution_blocked": resolution_blocked(n_train, alpha),
                "config_hash": config_hash,
                "lane": "research_review_candidate",
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "routing_uses_family_name": 0,
                "routing_uses_test_length": 0,
                "routing_uses_test_labels": 0,
                "routing_uses_test_positions": 0,
            }
            assert_label_free_prediction(row)
            rows.append(row)
    return rows


def assert_label_free_prediction(row: dict) -> None:
    forbidden = {
        "y_test",
        "test_labels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "hard_tp",
        "hard_fp",
        "hard_f1",
        "combined_f1",
    }
    overlap = forbidden & set(row)
    if overlap:
        raise ValueError(f"prediction contains evaluation fields: {sorted(overlap)}")
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    indices = [int(value) for value in row.get("candidate_indices", [])]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate candidate index")
    if any(value < 0 or value >= int(row["n_test"]) for value in indices):
        raise ValueError("candidate index out of TEST range")
    if len(indices) != int(row["candidate_count"]):
        raise ValueError("candidate count mismatch")


def evaluate_indices(indices: list[int], labels: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    selected = set(int(value) for value in indices)
    truth = set(np.flatnonzero(y == 1).astype(int).tolist())
    if any(value < 0 or value >= len(y) for value in selected):
        raise ValueError("candidate index out of label range")
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
