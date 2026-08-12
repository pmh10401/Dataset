"""Fixed adapters for four published classical one-class detectors."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM

from ..config import CLASSICAL_METHOD_IDS, METHOD_SPECS
from ..contracts import ScoreResult


def score_classical(
    method_id: str,
    fit: np.ndarray,
    query: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit one fixed detector and return query scores with anomaly pointing up."""
    fit_values, query_values = _validated_matrices(fit, query)
    if method_id not in CLASSICAL_METHOD_IDS:
        raise ValueError(f"unknown classical method: {method_id}")

    if method_id == "knn_distance":
        estimator = NearestNeighbors(n_neighbors=min(5, len(fit_values)))
        estimator.fit(fit_values)
        distances, _ = estimator.kneighbors(query_values)
        scores = distances.mean(axis=1)
    elif method_id == "lof_novelty":
        if len(fit_values) < 2:
            raise ValueError("LOF requires at least two fit rows")
        estimator = LocalOutlierFactor(
            n_neighbors=min(20, len(fit_values) - 1),
            novelty=True,
            contamination="auto",
            n_jobs=1,
        )
        estimator.fit(fit_values)
        scores = -estimator.score_samples(query_values)
    elif method_id == "ocsvm_rbf":
        estimator = OneClassSVM(
            kernel="rbf", gamma="scale", nu=0.01, cache_size=256
        )
        estimator.fit(fit_values)
        scores = -estimator.score_samples(query_values)
    else:
        estimator = IsolationForest(
            n_estimators=256,
            max_samples="auto",
            contamination="auto",
            random_state=int(seed),
            n_jobs=1,
        )
        estimator.fit(fit_values)
        scores = -estimator.score_samples(query_values)

    result = np.asarray(scores, dtype=np.float64).reshape(-1)
    if result.shape != (len(query_values),) or not np.isfinite(result).all():
        raise ValueError(f"{method_id} produced invalid scores")
    return result


def crossfit_scores(
    method_id: str,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> ScoreResult:
    """Create leakage-free TRAIN OOF reference scores and full-fit TEST scores."""
    train_values, test_values = _validated_matrices(train, test)
    minimum = _minimum_train_rows(method_id)
    if len(train_values) < minimum:
        return ScoreResult(
            method_id=method_id,
            status="abstain",
            reason="insufficient_train_rows",
            oof_scores=np.empty(0),
            test_scores=np.empty(0),
        )

    oof = np.empty(len(train_values), dtype=np.float64)
    splitter = KFold(
        n_splits=min(5, len(train_values)), shuffle=True, random_state=int(seed)
    )
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(train_values)):
        oof[held_indices] = score_classical(
            method_id,
            train_values[fit_indices],
            train_values[held_indices],
            seed=int(seed) + fold,
        )
    test_scores = score_classical(
        method_id, train_values, test_values, seed=int(seed)
    )
    return ScoreResult(
        method_id=method_id,
        status="complete",
        reason=None,
        oof_scores=oof,
        test_scores=test_scores,
    )


def _minimum_train_rows(method_id: str) -> int:
    for spec in METHOD_SPECS:
        if spec.method_id == method_id:
            return spec.min_train_rows
    raise ValueError(f"unknown classical method: {method_id}")


def _validated_matrices(
    fit: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    fit_values = np.asarray(fit, dtype=np.float64)
    query_values = np.asarray(query, dtype=np.float64)
    if fit_values.ndim != 2 or query_values.ndim != 2:
        raise ValueError("fit and query must be two-dimensional")
    if not len(fit_values) or not len(query_values):
        raise ValueError("fit and query must not be empty")
    if fit_values.shape[1] != query_values.shape[1]:
        raise ValueError("fit and query feature counts must match")
    if not np.isfinite(fit_values).all() or not np.isfinite(query_values).all():
        raise ValueError("fit and query must be finite")
    return fit_values, query_values
