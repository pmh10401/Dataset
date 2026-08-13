"""Frozen Task 2 registry for Exp452; no execution entry point is provided."""

from __future__ import annotations

from types import MappingProxyType


DATASET_COUNT_ANTIGRAVITY = 48
PREDICTION_COUNT_ANTIGRAVITY = 96
FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY = 16
EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY = "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d"
EXP449_PREDICTION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812"
EXP452_EVALUATION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp452_exp449_index_order_repair_evaluation_antigravity_20260813"
EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp452_index_order_repair_lifecycle_receipts_antigravity_20260813"
EXP452_FAILURE_AUDIT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp452_index_order_repair_failure_audit_antigravity.json"
INDEX_ORDER_RULE_ANTIGRAVITY = "sorted_ascending_exact_list"

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
    "experiment": "Exp452",
    "schema": "exp452",
    "kind": "exp449_index_order_repair",
    "run_id": "exp452-exp449-index-order-repair-antigravity-20260813",
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
    "predecessor_experiment": "Exp449",
    "predecessor_prediction_aggregate_sha256": EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY,
    "predecessor_prediction_root": EXP449_PREDICTION_ROOT_ANTIGRAVITY,
    "evaluation_root": EXP452_EVALUATION_ROOT_ANTIGRAVITY,
    "receipt_directory": EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY,
    "failure_audit": EXP452_FAILURE_AUDIT_ANTIGRAVITY,
    "target_dataset_count": DATASET_COUNT_ANTIGRAVITY,
    "target_prediction_count": PREDICTION_COUNT_ANTIGRAVITY,
    "target_family_method_pair_count": FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY,
    "index_order_rule": INDEX_ORDER_RULE_ANTIGRAVITY,
})
