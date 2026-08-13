"""Official CPU-only aeon RDST representation for Exp451 fixtures."""

from __future__ import annotations

import math

import numpy as np

from .config_antigravity import FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY


def _rows_antigravity(value: object, field: str, *, expected_length: int | None = None) -> np.ndarray:
    if type(value) is not np.ndarray or value.ndim != 3 or value.shape[0] == 0 or value.shape[1] != 1:
        raise ValueError(f"{field} must be a non-empty exact (rows, 1, length) ndarray")
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f"{field} must be numeric and finite")
    if expected_length is not None and value.shape[2] != expected_length:
        raise ValueError("RDST rows must retain the fitted length")
    # aeon 1.4.0's audited RDST kernel requires a contiguous float64 fixture.
    # This is input representation only; the official transformer remains unchanged.
    return np.ascontiguousarray(value, dtype=np.float64)


class RDSTRepresentationAntigravity:
    """Thin wrapper that preserves the audited public aeon implementation."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be an exact non-negative integer")
        self.seed = seed
        self._transformer: object | None = None
        self._length: int | None = None
        self._retained_shapelets: int | None = None

    def _new_transformer_antigravity(self) -> object:
        # This is intentionally a lazy import: RDST itself is never an accelerator target.
        from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
        return RandomDilatedShapeletTransform(max_shapelets=1024, random_state=self.seed, n_jobs=1)

    @property
    def retained_shapelets(self) -> int:
        if self._retained_shapelets is None:
            raise RuntimeError("RDST must be fit before retained_shapelets is available")
        return self._retained_shapelets

    def fit(self, rows: object, y: object = None) -> "RDSTRepresentationAntigravity":
        if y is not None:
            raise ValueError("official RDST must receive y=None only")
        values = _rows_antigravity(rows, "RDST fit rows")
        transformer = self._new_transformer_antigravity()
        transformer.fit(values, y=None)  # type: ignore[attr-defined]
        retained = getattr(transformer, "n_shapelets_", None)
        if type(retained) is not int or retained < 1:
            raise RuntimeError("official RDST did not expose a positive n_shapelets_")
        self._transformer = transformer
        self._length = int(values.shape[2])
        self._retained_shapelets = retained
        return self

    def transform(self, rows: object) -> np.ndarray:
        if self._transformer is None or self._length is None:
            raise ValueError("RDST transform requires a fitted representation")
        values = _rows_antigravity(rows, "RDST transform rows", expected_length=self._length)
        features = np.asarray(self._transformer.transform(values), dtype=float)  # type: ignore[attr-defined]
        if features.ndim != 2 or features.shape != (len(values), 3 * self.retained_shapelets) or not np.isfinite(features).all():
            raise ValueError("official RDST returned non-finite or incompatible features")
        return features

    def fit_transform(self, rows: object, y: object = None) -> np.ndarray:
        return self.fit(rows, y=y).transform(rows)


def make_rdst_antigravity(seed: int) -> RDSTRepresentationAntigravity:
    return RDSTRepresentationAntigravity(seed)
