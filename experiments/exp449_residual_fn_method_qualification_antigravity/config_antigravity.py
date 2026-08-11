"""Frozen Task 2 registry for Exp449; no execution entry point is provided."""

from __future__ import annotations

from types import MappingProxyType


FIXED_INTERPRETER_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"

CANDIDATE_MATRIX_ANTIGRAVITY = (
    ("Crop", ("minirocket_local_gap", "multirocket_local_gap")),
    ("ElectricDevices", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
    ("FordA", ("minirocket_local_gap", "hydra_local_gap")),
    ("FordB", ("multirocket_local_gap", "hydra_local_gap")),
    ("Yoga", ("multirocket_local_gap", "hydra_multirocket_local_gap")),
    ("ECG5000", ("multirocket_local_gap", "hydra_local_gap")),
    ("ChlorineConcentration", ("minirocket_local_gap", "multirocket_local_gap")),
    ("StarLightCurves", ("multirocket_local_gap", "hydra_local_gap")),
)

FROZEN_CONFIG_ANTIGRAVITY = MappingProxyType({
    "experiment": "Exp449",
    "schema": "exp449",
    "kind": "residual_fn_method_qualification",
    "run_id": "exp449-residual-fn-method-qualification-antigravity-20260812",
    "fixed_interpreter": FIXED_INTERPRETER_ANTIGRAVITY,
    "python_version": "3.14.6",
    "aeon_version": "1.4.0",
    "numpy_version": "2.3.5",
    "sklearn_version": "1.7.2",
    "seeds": (20260721, 20261721),
    "alphas": (0.005,),
    "seed_aggregation": "elementwise_median",
    "calibration": "empirical_upper_pvalue_lte_alpha",
    "crossfit_folds": 5,
    "knn_neighbors": 3,
    "max_train_reference": 1000,
    "query_chunk_size": 128,
    "minirocket_requested_kernels": 4096,
    "multirocket_requested_kernels": 1024,
    "multirocket_max_dilations_per_kernel": 32,
    "multirocket_features_per_kernel": 4,
    "multirocket_normalise": False,
    "hydra_kernels": 8,
    "hydra_groups": 64,
    "global_fn_must_be_below": 1988,
    "global_fp_max": 314,
    "global_f1_min": 0.6264199935,
    "candidate_matrix": CANDIDATE_MATRIX_ANTIGRAVITY,
})
