"""Exact Exp249 LocalGap semantics under the Exp449 namespace."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


class LocalGapModelAntigravity:
    def __init__(self, reference: np.ndarray, *, neighbors: int = 3) -> None:
        values = np.asarray(reference, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("local-gap requires a two-dimensional reference")
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(values)
        scaled = self.scaler.transform(values)
        self.neighbors = max(1, min(int(neighbors), len(scaled)))
        self.reference_model = NearestNeighbors(n_neighbors=max(1, min(self.neighbors + 1, len(scaled))), metric="euclidean").fit(scaled)
        distances, indices = self.reference_model.kneighbors(scaled)
        self.local_radius = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        self.query_model = NearestNeighbors(n_neighbors=self.neighbors, metric="euclidean").fit(scaled)
        self.scale = float(np.median(self.local_radius) + 1e-9)

    def score(self, query: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(query, dtype=float))
        distances, indices = self.query_model.kneighbors(scaled)
        base = distances.mean(axis=1)
        local = self.local_radius[indices].mean(axis=1) + 1e-9
        return np.asarray((base - local) / self.scale, dtype=float)
