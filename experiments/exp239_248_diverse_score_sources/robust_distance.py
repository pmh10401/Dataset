from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import RobustScaler


class RobustDistanceModel:
    def __init__(self, reference: np.ndarray):
        values = np.asarray(reference, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("robust distance requires a two-dimensional reference")
        self.scaler = RobustScaler(quantile_range=(10, 90)).fit(values)
        scaled = self.scaler.transform(values)
        self.center = np.median(scaled, axis=0)
        if scaled.shape[1] == 1:
            variance = float(np.var(scaled[:, 0]) + 1e-6)
            self.precision = np.asarray([[1.0 / variance]])
        else:
            self.precision = LedoitWolf().fit(scaled).precision_

    def score(self, query: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(query, dtype=float))
        delta = scaled - self.center
        squared = np.einsum("ij,jk,ik->i", delta, self.precision, delta)
        return np.sqrt(np.maximum(squared, 0.0))


def robust_distance(
    train_reference: np.ndarray, query: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    model = RobustDistanceModel(train_reference)
    return model.score(train_reference), model.score(query)
