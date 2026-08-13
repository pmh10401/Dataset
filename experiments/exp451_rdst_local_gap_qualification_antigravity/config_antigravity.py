"""Immutable scientific configuration for the planned Exp451 RDST qualification."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType


FIXED_INTERPRETER_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"


@dataclass(frozen=True, slots=True)
class FrozenConfigAuthorityAntigravity:
    experiment: str; schema: str; kind: str; run_id: str; fixed_interpreter: str
    python_version: str; aeon_version: str; numpy_version: str; sklearn_version: str
    rdst_class_path: str; source_sha256: str; max_shapelets: int; seeds: tuple[int, int]
    alpha: float; rdst_n_jobs: int; fit_label_policy: str; seed_aggregation: str
    crossfit_folds: int; knn_neighbors: int; max_train_reference: int; query_chunk_size: int
    target_family_count: int; target_dataset_count: int; target_test_rows: int
    reuse_dataset_count: int; baseline_tp: int; baseline_fp: int; baseline_fn: int
    baseline_tn: int; baseline_f1: float; worker_rule: str; rdst_thread_rule: str
    prediction_root: str; evaluation_root: str


FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY = FrozenConfigAuthorityAntigravity(
    "Exp451", "exp451", "rdst_local_gap_qualification", "exp451-rdst-local-gap-qualification-antigravity-20260813",
    FIXED_INTERPRETER_ANTIGRAVITY, "3.14.6", "1.4.0", "2.3.5", "1.7.2",
    "aeon.transformations.collection.shapelet_based._dilated_shapelet_transform.RandomDilatedShapeletTransform",
    "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2", 1024,
    (20260721, 20261721), 0.005, 1, "y_none_only", "elementwise_median", 5, 3, 1000, 128,
    8, 48, 72300, 1069, 1930, 314, 1988, 191668, 0.6264199935,
    "max(1, logical_cpus-1)", "n_jobs=1",
    "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_antigravity_20260813",
    "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_evaluation_antigravity_20260813",
)
FROZEN_CONFIG_ANTIGRAVITY = MappingProxyType(asdict(FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY))


def _exact_str_antigravity(value: object, field: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be an exact non-empty string")


def _exact_int_antigravity(value: object, field: str, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer at least {minimum}")


def _exact_float_antigravity(value: object, field: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be an exact finite float")


def decode_frozen_config_antigravity(value: object) -> FrozenConfigAuthorityAntigravity:
    """Decode only the exact preregistered mapping; no coercion or silent defaults."""
    names = {field.name for field in fields(FrozenConfigAuthorityAntigravity)}
    if type(value) is not dict or set(value) != names:
        raise ValueError("frozen config keys differ")
    for field in ("experiment", "schema", "kind", "run_id", "fixed_interpreter", "python_version", "aeon_version", "numpy_version", "sklearn_version", "rdst_class_path", "source_sha256", "fit_label_policy", "seed_aggregation", "worker_rule", "rdst_thread_rule", "prediction_root", "evaluation_root"):
        _exact_str_antigravity(value[field], field)
    for field in ("max_shapelets", "rdst_n_jobs", "crossfit_folds", "knn_neighbors", "max_train_reference", "query_chunk_size", "target_family_count", "target_dataset_count", "target_test_rows", "reuse_dataset_count", "baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn"):
        _exact_int_antigravity(value[field], field, minimum=1 if field not in {"baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn"} else 0)
    _exact_float_antigravity(value["alpha"], "alpha")
    _exact_float_antigravity(value["baseline_f1"], "baseline_f1")
    seeds = value["seeds"]
    if type(seeds) is not tuple or len(seeds) != 2:
        raise ValueError("seeds must be an exact two-item tuple")
    for seed in seeds:
        _exact_int_antigravity(seed, "seed")
    if not 0.0 < value["alpha"] <= 1.0 or not 0.0 < value["baseline_f1"] <= 1.0 or not re.fullmatch(r"[0-9a-f]{64}", value["source_sha256"]):
        raise ValueError("frozen config values are invalid")
    authority = FrozenConfigAuthorityAntigravity(**value)
    if authority != FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY:
        raise ValueError("frozen config values differ")
    return authority
