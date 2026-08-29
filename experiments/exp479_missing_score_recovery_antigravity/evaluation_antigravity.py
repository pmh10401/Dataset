"""Four-method label-separated performance evaluation for Exp479."""

from __future__ import annotations

import math
import sqlite3
from typing import Mapping, TYPE_CHECKING

import numpy as np
from sklearn.metrics import roc_auc_score

from .config_antigravity import (
    ALPHA_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    NUMBA_CACHE_DIR_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
)
from .contracts_antigravity import PreparedDatasetAntigravity

if TYPE_CHECKING:
    from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
        CandidateScoresAntigravity,
    )


_LABEL_SQL_ANTIGRAVITY = (
    "SELECT i.instance_index, i.label FROM instances AS i "
    "JOIN datasets AS d ON d.id=i.dataset_id "
    "WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index"
)


def score_prepared_methods_antigravity(
    prepared: PreparedDatasetAntigravity,
) -> dict[str, "CandidateScoresAntigravity"]:
    """Score all four frozen methods without accepting or opening TEST labels."""
    if type(prepared) is not PreparedDatasetAntigravity:
        raise ValueError("prepared must be an exact PreparedDatasetAntigravity")
    from numba import config as numba_config

    NUMBA_CACHE_DIR_ANTIGRAVITY.mkdir(parents=True, exist_ok=True)
    numba_config.CACHE_DIR = str(NUMBA_CACHE_DIR_ANTIGRAVITY)
    from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
        MethodCandidateAntigravity,
    )
    from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
        CandidateScoresAntigravity,
        score_candidate_antigravity,
    )

    results: dict[str, CandidateScoresAntigravity] = {}
    for method in METHODS_ANTIGRAVITY:
        candidate = MethodCandidateAntigravity(
            method,
            method,
            (SEED_ANTIGRAVITY,),
            (ALPHA_ANTIGRAVITY,),
        )
        results[method] = score_candidate_antigravity(
            prepared.train_rows,
            prepared.test_rows,
            candidate,
            seed=SEED_ANTIGRAVITY,
        )
    return results


def load_retained_test_labels_antigravity(
    connection: sqlite3.Connection,
    dataset: str,
    retained_indices: tuple[int, ...],
) -> np.ndarray:
    """Open TEST labels only after scoring and align them to finite retained rows."""
    if type(connection) is not sqlite3.Connection:
        raise ValueError("connection must be an exact sqlite3.Connection")
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset must be an exact non-empty string")
    if type(retained_indices) is not tuple or any(
        type(index) is not int or index < 0 for index in retained_indices
    ):
        raise ValueError("retained_indices must be an exact tuple")
    labels_by_index: dict[int, int] = {}
    for index, raw_label in connection.execute(_LABEL_SQL_ANTIGRAVITY, (dataset,)):
        if type(index) is not int or type(raw_label) is not str or raw_label not in ("0", "1"):
            raise ValueError("TEST labels must be binary 0/1 rows")
        labels_by_index[index] = int(raw_label)
    try:
        values = np.asarray(
            [labels_by_index[index] for index in retained_indices], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("a retained TEST index has no label") from error
    return np.frombuffer(values.tobytes(), dtype=np.int64)


def evaluate_score_vector_antigravity(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, object]:
    """Compute frozen AUC and strict test-normal-max zero-FP true positives."""
    scores = np.asarray(test_scores, dtype=np.float64)
    labels = np.asarray(test_labels, dtype=np.int64)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels) or not len(scores):
        raise ValueError("aligned non-empty one-dimensional scores and labels are required")
    if not np.isfinite(scores).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("finite scores and both binary TEST classes are required")
    normal_max = float(scores[labels == 0].max())
    auc = float(roc_auc_score(labels, scores))
    if not math.isfinite(auc):
        raise ValueError("AUC must be finite")
    return {
        "auc": auc,
        "zf": int(np.count_nonzero(scores[labels == 1] > normal_max)),
    }


def evaluate_scored_dataset_antigravity(
    dataset: str,
    scores_by_method: Mapping[str, np.ndarray],
    test_labels: np.ndarray,
    official_method: str,
) -> dict[str, object]:
    """Evaluate completed score vectors after the label boundary opens."""
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset must be an exact non-empty string")
    if tuple(scores_by_method) != METHODS_ANTIGRAVITY:
        raise ValueError("scores_by_method must contain the four frozen methods in order")
    if official_method not in METHODS_ANTIGRAVITY:
        raise ValueError("official_method is not a frozen method")
    metrics = {
        method: evaluate_score_vector_antigravity(scores_by_method[method], test_labels)
        for method in METHODS_ANTIGRAVITY
    }
    baseline_method = METHODS_ANTIGRAVITY[0]
    return {
        "dataset": dataset,
        "scores": metrics,
        "official_method": official_method,
        "baseline_method": baseline_method,
        "official": metrics[official_method],
        "baseline": metrics[baseline_method],
        "oracle_auc": max(float(value["auc"]) for value in metrics.values()),
    }


__all__ = [
    "evaluate_score_vector_antigravity",
    "evaluate_scored_dataset_antigravity",
    "load_retained_test_labels_antigravity",
    "score_prepared_methods_antigravity",
]
