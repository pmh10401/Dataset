"""Immutable scientific configuration for the planned Exp451 RDST qualification."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


FIXED_INTERPRETER_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"

_FROZEN_CONFIG_DATA_ANTIGRAVITY = {
    "experiment": "Exp451",
    "schema": "exp451",
    "kind": "rdst_local_gap_qualification",
    "run_id": "exp451-rdst-local-gap-qualification-antigravity-20260813",
    "fixed_interpreter": FIXED_INTERPRETER_ANTIGRAVITY,
    "python_version": "3.14.6",
    "aeon_version": "1.4.0",
    "numpy_version": "2.3.5",
    "sklearn_version": "1.7.2",
    "rdst_class_path": "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform",
    "source_sha256": "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2",
    "max_shapelets": 1024,
    "seeds": (20260721, 20261721),
    "alpha": 0.005,
    "rdst_n_jobs": 1,
    "fit_label_policy": "y_none_only",
    "seed_aggregation": "elementwise_median",
    "crossfit_folds": 5,
    "knn_neighbors": 3,
    "max_train_reference": 1000,
    "query_chunk_size": 128,
    "target_family_count": 8,
    "target_dataset_count": 48,
    "target_test_rows": 72300,
    "reuse_dataset_count": 1069,
    "baseline_tp": 1930,
    "baseline_fp": 314,
    "baseline_fn": 1988,
    "baseline_tn": 191668,
    "baseline_f1": 0.6264199935,
    "worker_rule": "max(1, logical_cpus-1)",
    "rdst_thread_rule": "n_jobs=1",
    "prediction_root": "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_antigravity_20260813",
    "evaluation_root": "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_evaluation_antigravity_20260813",
}

FROZEN_CONFIG_ANTIGRAVITY = MappingProxyType(_FROZEN_CONFIG_DATA_ANTIGRAVITY)


def _exact_str_antigravity(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be an exact non-empty string")
    return value


def _exact_int_antigravity(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer at least {minimum}")
    return value


def _exact_float_antigravity(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be an exact finite float")
    return value


@dataclass(frozen=True, slots=True)
class FrozenConfigAuthorityAntigravity:
    experiment: str
    schema: str
    kind: str
    run_id: str
    fixed_interpreter: str
    python_version: str
    aeon_version: str
    numpy_version: str
    sklearn_version: str
    rdst_class_path: str
    source_sha256: str
    max_shapelets: int
    seeds: tuple[int, int]
    alpha: float
    rdst_n_jobs: int
    fit_label_policy: str
    seed_aggregation: str
    crossfit_folds: int
    knn_neighbors: int
    max_train_reference: int
    query_chunk_size: int
    target_family_count: int
    target_dataset_count: int
    target_test_rows: int
    reuse_dataset_count: int
    baseline_tp: int
    baseline_fp: int
    baseline_fn: int
    baseline_tn: int
    baseline_f1: float
    worker_rule: str
    rdst_thread_rule: str
    prediction_root: str
    evaluation_root: str


def decode_frozen_config_antigravity(value: object) -> FrozenConfigAuthorityAntigravity:
    """Decode only the exact preregistered mapping; no coercion or silent defaults."""
    if type(value) is not dict or set(value) != set(_FROZEN_CONFIG_DATA_ANTIGRAVITY):
        raise ValueError("frozen config keys differ")
    strings = (
        "experiment", "schema", "kind", "run_id", "fixed_interpreter", "python_version",
        "aeon_version", "numpy_version", "sklearn_version", "rdst_class_path", "source_sha256",
        "fit_label_policy", "seed_aggregation", "worker_rule", "rdst_thread_rule", "prediction_root",
        "evaluation_root",
    )
    for field in strings:
        _exact_str_antigravity(value[field], field)
    integers = (
        "max_shapelets", "rdst_n_jobs", "crossfit_folds", "knn_neighbors", "max_train_reference",
        "query_chunk_size", "target_family_count", "target_dataset_count", "target_test_rows",
        "reuse_dataset_count", "baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn",
    )
    for field in integers:
        _exact_int_antigravity(value[field], field, minimum=1 if field not in {"baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn"} else 0)
    alpha = _exact_float_antigravity(value["alpha"], "alpha")
    baseline_f1 = _exact_float_antigravity(value["baseline_f1"], "baseline_f1")
    seeds = value["seeds"]
    if type(seeds) is not tuple or len(seeds) != 2:
        raise ValueError("seeds must be an exact two-item tuple")
    for seed in seeds:
        _exact_int_antigravity(seed, "seed", minimum=0)
    if not 0.0 < alpha <= 1.0 or not 0.0 < baseline_f1 <= 1.0:
        raise ValueError("frozen probability is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]):
        raise ValueError("source_sha256 must be a lowercase SHA256")
    authority = FrozenConfigAuthorityAntigravity(**value)
    if authority != FrozenConfigAuthorityAntigravity(**_FROZEN_CONFIG_DATA_ANTIGRAVITY):
        raise ValueError("frozen config values differ")
    return authority


FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY = decode_frozen_config_antigravity(dict(FROZEN_CONFIG_ANTIGRAVITY))
