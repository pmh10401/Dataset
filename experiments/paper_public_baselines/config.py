"""Frozen scientific configuration for the public baseline comparison."""

from __future__ import annotations

from dataclasses import dataclass


ALPHAS = (0.005, 0.01, 0.02)
SEED = 20260812
EXPECTED_DATASETS = 1117
EXPECTED_DB_SHA256 = (
    "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
)
CLASSICAL_METHOD_IDS = (
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
)
DEEP_METHOD_IDS = (
    "deep_svdd_1d",
    "drocc_1d",
)
METHOD_IDS = CLASSICAL_METHOD_IDS + DEEP_METHOD_IDS


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    family: str
    min_train_rows: int


METHOD_SPECS = (
    MethodSpec("knn_distance", "distance", 2),
    MethodSpec("lof_novelty", "local_density", 3),
    MethodSpec("ocsvm_rbf", "boundary", 2),
    MethodSpec("isolation_forest", "isolation", 2),
    MethodSpec("deep_svdd_1d", "deep_boundary", 8),
    MethodSpec("drocc_1d", "deep_robust", 8),
)


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    alphas: tuple[float, ...] = ALPHAS
    seed: int = SEED
    expected_datasets: int = EXPECTED_DATASETS
    expected_db_sha256: str = EXPECTED_DB_SHA256
    methods: tuple[MethodSpec, ...] = METHOD_SPECS


FROZEN_CONFIG = FrozenConfig()

