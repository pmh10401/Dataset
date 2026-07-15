from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.exp165.similarity_calibration_core import (
    SOURCES,
    higher_threshold,
    robust_location_scale,
)


ALPHAS = (0.005, 0.01, 0.02, 0.05)
METHODS = ("local_crossfit_reference", "similarity_pool_reference")
AGREEMENTS = (2, 3)


def normalize_query_scores(
    train_oof_scores: dict[str, np.ndarray],
    query_scores: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    normalized_train: dict[str, np.ndarray] = {}
    normalized_query: dict[str, np.ndarray] = {}
    for source in SOURCES:
        location, scale = robust_location_scale(train_oof_scores[source])
        normalized_train[source] = (
            np.asarray(train_oof_scores[source], dtype=np.float64) - location
        ) / scale
        normalized_query[source] = (
            np.asarray(query_scores[source], dtype=np.float64) - location
        ) / scale
    return normalized_train, normalized_query


def candidate_rows(
    *,
    dataset: str,
    n_train: int,
    n_test: int,
    normalized_test_scores: dict[str, np.ndarray],
    references: dict[str, dict[str, np.ndarray]],
    selected_donors: list[str],
    config_hash: str,
) -> list[dict]:
    rows: list[dict] = []
    for method in METHODS:
        source_references = references[method]
        for alpha in ALPHAS:
            thresholds = {
                source: higher_threshold(source_references[source], alpha)
                for source in SOURCES
            }
            flags = {
                source: np.asarray(normalized_test_scores[source] > thresholds[source], dtype=bool)
                for source in SOURCES
            }
            source_count = np.vstack([flags[source] for source in SOURCES]).sum(axis=0)
            for agreement in AGREEMENTS:
                indices = np.flatnonzero(source_count >= agreement).astype(int).tolist()
                row = {
                    "dataset": dataset,
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "method": method,
                    "alpha": float(alpha),
                    "agreement": f"{agreement}of3",
                    "candidate_indices": indices,
                    "candidate_count": len(indices),
                    "selected_donor_count": len(selected_donors) if method == "similarity_pool_reference" else 0,
                    "selected_donors": selected_donors if method == "similarity_pool_reference" else [],
                    "reference_counts": {
                        source: int(np.asarray(source_references[source]).size) for source in SOURCES
                    },
                    "thresholds": thresholds,
                    "source_candidate_counts": {
                        source: int(flags[source].sum()) for source in SOURCES
                    },
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
        "y_test", "test_labels", "tp", "fp", "fn", "precision", "recall",
        "f1", "hard_tp", "hard_fp", "hard_f1", "combined_f1",
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
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def stable_config_hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
