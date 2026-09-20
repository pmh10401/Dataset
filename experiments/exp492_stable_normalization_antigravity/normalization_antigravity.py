"""Train-only near-constant guards for official SparseScaler and RobustScaler."""
from __future__ import annotations

import numpy as np
import torch
from aeon.regression.convolution_based._hydra import _SparseScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from ..exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity

_EPS32_ANTIGRAVITY = float(np.finfo(np.float32).eps)


def _as_1d_numpy_antigravity(values: object, dtype) -> np.ndarray:
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    return np.asarray(array, dtype=dtype).reshape(-1)


class StableSparseScalerAntigravity(_SparseScaler):
    def fit(self, X, y=None):
        tensor = X if torch.is_tensor(X) else torch.as_tensor(X)
        if tensor.ndim != 2 or tensor.shape[0] == 0:
            raise ValueError("sparse scaler requires a non-empty two-dimensional array")
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite sparse scaler input")
        super().fit(tensor)
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.sigma).all():
            raise ValueError("non-finite sparse scaler statistics")
        original_scale = self.sigma.detach().clone()
        original_spread = tensor.clamp(0).sqrt().std(0)
        tolerance = (10.0 * _EPS32_ANTIGRAVITY) * torch.maximum(torch.ones_like(self.mu), self.mu.abs())
        degeneracy_mask = original_scale <= tolerance
        changed_mask = degeneracy_mask & (original_scale < 1)
        self.sigma = torch.where(changed_mask, torch.ones_like(self.sigma), self.sigma)
        self.original_spread_ = _as_1d_numpy_antigravity(original_spread, float)
        self.original_scale_ = _as_1d_numpy_antigravity(original_scale, float)
        self.tolerance_ = _as_1d_numpy_antigravity(tolerance, float)
        self.degeneracy_mask_ = _as_1d_numpy_antigravity(degeneracy_mask, bool)
        self.changed_mask_ = _as_1d_numpy_antigravity(changed_mask, bool)
        return self


class StableRobustScalerAntigravity(RobustScaler):
    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("robust scaler requires a non-empty two-dimensional array")
        if not np.isfinite(values).all():
            raise ValueError("non-finite robust scaler input")
        super().fit(X, y)
        q10, q90 = np.percentile(values, [10, 90], axis=0, method="linear")
        original_spread = np.asarray(q90 - q10, dtype=float).reshape(-1)
        tolerance = (10.0 * _EPS32_ANTIGRAVITY) * np.maximum(1.0, np.maximum(np.abs(q10), np.abs(q90)))
        tolerance = np.asarray(tolerance, dtype=float).reshape(-1)
        if not np.isfinite(original_spread).all() or not np.isfinite(tolerance).all() or (self.center_ is not None and not np.isfinite(self.center_).all()):
            raise ValueError("non-finite robust scaler statistics")
        degeneracy_mask = original_spread <= tolerance
        if self.scale_ is None:
            original_scale = np.ones(original_spread.shape[0], dtype=float)
            changed_mask = np.zeros(original_spread.shape[0], dtype=bool)
        else:
            original_scale = np.asarray(self.scale_, dtype=float).reshape(-1).copy()
            if not np.isfinite(original_scale).all():
                raise ValueError("non-finite robust scaler scale")
            changed_mask = degeneracy_mask & (original_scale < 1.0)
            self.scale_ = np.where(changed_mask, 1.0, original_scale)
        self.original_spread_ = original_spread
        self.original_scale_ = original_scale
        self.tolerance_ = tolerance
        self.degeneracy_mask_ = np.asarray(degeneracy_mask, dtype=bool).reshape(-1)
        self.changed_mask_ = np.asarray(changed_mask, dtype=bool).reshape(-1)
        return self


class StableLocalGapModelAntigravity(LocalGapModelAntigravity):
    def __init__(self, reference: np.ndarray, *, neighbors: int = 3) -> None:
        values = np.asarray(reference, dtype=float)
        if values.ndim != 2 or len(values) < 2:
            raise ValueError("local-gap requires a two-dimensional reference")
        if not np.isfinite(values).all():
            raise ValueError("non-finite local-gap reference")
        self.scaler = StableRobustScalerAntigravity(quantile_range=(10, 90)).fit(values)
        scaled = self.scaler.transform(values)
        if not np.isfinite(scaled).all():
            raise ValueError("non-finite local-gap scaled reference")
        self.neighbors = max(1, min(int(neighbors), len(scaled)))
        self.reference_model = NearestNeighbors(n_neighbors=max(1, min(self.neighbors + 1, len(scaled))), metric="euclidean").fit(scaled)
        distances, indices = self.reference_model.kneighbors(scaled)
        self.local_radius = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]
        self.query_model = NearestNeighbors(n_neighbors=self.neighbors, metric="euclidean").fit(scaled)
        self.scale = float(np.median(self.local_radius) + 1e-9)

    def score(self, query: np.ndarray) -> np.ndarray:
        values = np.asarray(query, dtype=float)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("non-finite local-gap query")
        result = super().score(values)
        if not np.isfinite(result).all():
            raise ValueError("non-finite local-gap scores")
        return result
