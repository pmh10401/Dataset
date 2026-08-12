"""Frozen robust scaling and local-gap KNN primitives.

Provenance: run_balanced_improvement_experiment.py SHA256
d784f8dcc3a5f05ef7d96829acbe9b9bb206ed5f37db0a5434bd848657f48efd
(`density_knn_score_pair`) and run_model_hard_research_experiments.py SHA256
f9f06973009259590b4afd165dc275cff9b834a628ccb17e248d015d55680194
(`scale_feature_pair`).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


PROVENANCE_PAPER = {
    "run_balanced_improvement_experiment.py": ("density_knn_score_pair",),
    "run_model_hard_research_experiments.py": ("scale_feature_pair",),
}


def scale_feature_pair_paper(train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(10, 90))
    return (scaler.fit_transform(train_features).astype(np.float32),
            scaler.transform(test_features).astype(np.float32))


def density_knn_score_pair_paper(train_features: np.ndarray, test_features: np.ndarray,
                                 neighbors: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)
    n_train = len(train_scaled)
    if n_train == 0:
        raise ValueError("density KNN requires nonempty TRAIN features")
    k_self = max(1, min(int(neighbors) + 1, n_train))
    train_nn = NearestNeighbors(n_neighbors=k_self, metric="euclidean").fit(train_scaled)
    train_distances, train_indices = train_nn.kneighbors(train_scaled)
    if train_distances.shape[1] > 1:
        base_train, neighbor_indices = train_distances[:, 1:].mean(axis=1), train_indices[:, 1:]
    else:
        base_train, neighbor_indices = train_distances[:, 0], train_indices
    local_radius = base_train + 1e-9
    local_train = np.mean(local_radius[neighbor_indices], axis=1) + 1e-9
    test_nn = NearestNeighbors(n_neighbors=max(1, min(int(neighbors), n_train)), metric="euclidean").fit(train_scaled)
    test_distances, test_indices = test_nn.kneighbors(test_scaled)
    base_test = test_distances.mean(axis=1)
    local_test = np.mean(local_radius[test_indices], axis=1) + 1e-9
    if mode == "ratio":
        return base_train / local_train, base_test / local_test
    if mode == "local_gap":
        scale = np.median(local_radius) + 1e-9
        return (base_train - local_train) / scale, (base_test - local_test) / scale
    raise ValueError(f"Unknown density KNN mode: {mode}")
