from __future__ import annotations

import numpy as np
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

from experiments.exp249_250_official_mini_multirocket.config import RocketModelConfig


def actual_kernel_count(requested_kernels: int) -> int:
    requested = int(requested_kernels)
    if requested < 84:
        raise ValueError("official ROCKET transforms require at least 84 kernels")
    return requested - requested % 84


def expected_feature_dimension(config: RocketModelConfig, requested_kernels: int) -> int:
    kernels = actual_kernel_count(requested_kernels)
    if config.transformer == "minirocket":
        return kernels
    if config.transformer == "multirocket":
        return 2 * int(config.n_features_per_kernel) * kernels
    raise ValueError(f"unknown transformer: {config.transformer}")


def as_aeon_collection(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("aligned series must be a two-dimensional matrix")
    if values.shape[1] < 9:
        values = np.pad(values, ((0, 0), (0, 9 - values.shape[1])))
    return values[:, np.newaxis, :]


def make_transformer(
    config: RocketModelConfig, *, requested_kernels: int, seed: int
) -> MiniRocket | MultiRocket:
    common = {
        "n_kernels": int(requested_kernels),
        "max_dilations_per_kernel": int(config.max_dilations_per_kernel),
        "n_jobs": 1,
        "random_state": int(seed),
    }
    if config.transformer == "minirocket":
        return MiniRocket(**common)
    if config.transformer == "multirocket":
        return MultiRocket(
            **common,
            n_features_per_kernel=int(config.n_features_per_kernel),
            normalise=bool(config.normalise),
        )
    raise ValueError(f"unknown transformer: {config.transformer}")

