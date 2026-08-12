"""Fixed CPU aeon representation adapters for label-free Exp449 scoring."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from aeon.regression.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import HydraTransformer, MiniRocket, MultiRocket
from sklearn.preprocessing import StandardScaler


class RepresentationAntigravity(Protocol):
    def fit_transform(self, rows: np.ndarray) -> np.ndarray: ...

    def transform(self, rows: np.ndarray) -> np.ndarray: ...


_REPRESENTATIONS_ANTIGRAVITY = frozenset({
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
})


def _rows_antigravity(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("rows must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("rows must be finite")
    if values.shape[1] < 9:
        values = np.pad(values, ((0, 0), (0, 9 - values.shape[1])))
    return values[:, np.newaxis, :]


def _features_antigravity(values: object) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("representation produced non-finite two-dimensional features")
    return result


class _SingleRepresentationAntigravity:
    def __init__(self, transformer: object) -> None:
        self._transformer = transformer

    def fit_transform(self, rows: np.ndarray) -> np.ndarray:
        return _features_antigravity(self._transformer.fit_transform(_rows_antigravity(rows)))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return _features_antigravity(self._transformer.transform(_rows_antigravity(rows)))


class _HydraMultiRocketRepresentationAntigravity:
    def __init__(self, seed: int) -> None:
        self._hydra = HydraTransformer(n_kernels=8, n_groups=64, n_jobs=1, random_state=seed)
        self._multirocket = MultiRocket(
            n_kernels=1024,
            max_dilations_per_kernel=32,
            n_features_per_kernel=4,
            normalise=False,
            n_jobs=1,
            random_state=seed,
        )
        self._hydra_scaler = _SparseScaler()
        self._multirocket_scaler = StandardScaler()

    def fit_transform(self, rows: np.ndarray) -> np.ndarray:
        collection = _rows_antigravity(rows)
        hydra = self._hydra_scaler.fit_transform(self._hydra.fit_transform(collection))
        multirocket = self._multirocket_scaler.fit_transform(self._multirocket.fit_transform(collection))
        return _features_antigravity(np.concatenate((_features_antigravity(hydra), _features_antigravity(multirocket)), axis=1))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        collection = _rows_antigravity(rows)
        hydra = self._hydra_scaler.transform(self._hydra.transform(collection))
        multirocket = self._multirocket_scaler.transform(self._multirocket.transform(collection))
        return _features_antigravity(np.concatenate((_features_antigravity(hydra), _features_antigravity(multirocket)), axis=1))


def make_representation_antigravity(name: str, seed: int) -> RepresentationAntigravity:
    """Construct one frozen CPU aeon feature source; no classifier is instantiated."""
    if type(name) is not str or name not in _REPRESENTATIONS_ANTIGRAVITY:
        raise ValueError("representation is not a fixed Exp449 source")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    if name == "minirocket_local_gap":
        return _SingleRepresentationAntigravity(MiniRocket(n_kernels=4096, n_jobs=1, random_state=seed))
    if name == "multirocket_local_gap":
        return _SingleRepresentationAntigravity(MultiRocket(n_kernels=1024, max_dilations_per_kernel=32, n_features_per_kernel=4, normalise=False, n_jobs=1, random_state=seed))
    if name == "hydra_local_gap":
        return _SingleRepresentationAntigravity(HydraTransformer(n_kernels=8, n_groups=64, n_jobs=1, random_state=seed))
    return _HydraMultiRocketRepresentationAntigravity(seed)
