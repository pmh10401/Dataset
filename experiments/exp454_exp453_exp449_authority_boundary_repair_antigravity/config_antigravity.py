"""Frozen Task 2 registry for Exp454; no execution entry point is provided."""

from __future__ import annotations

from types import MappingProxyType


DATASET_COUNT_ANTIGRAVITY = 48
PREDICTION_COUNT_ANTIGRAVITY = 96
FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY = 16
EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY = "a6054a44fc4e78f354dcf61d2e003284804680cd4197405364fcb5ef8d822e0d"
EXP449_PREDICTION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812"
EXP454_EVALUATION_ROOT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp454_exp453_exp449_authority_boundary_repair_evaluation_antigravity_20260814"
EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp454_authority_boundary_repair_lifecycle_receipts_antigravity_20260814"
EXP454_FAILURE_AUDIT_ANTIGRAVITY = "/Users/mac/Documents/Dataset/exp454_authority_boundary_repair_failure_audit_antigravity.json"
INDEX_ORDER_RULE_ANTIGRAVITY = "sorted_ascending_exact_list"

FIXED_INTERPRETER_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
FIXED_EXECUTABLE_ANTIGRAVITY = FIXED_INTERPRETER_ANTIGRAVITY
FIXED_PREFIX_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
RUNTIME_VERSIONS_ANTIGRAVITY = ("3.14.6", "2.3.5", "1.4.0", "1.7.2")
EVALUATION_LEAVES_ANTIGRAVITY = (
    "context_antigravity.json", "compatibility_antigravity.json", "candidate_registry_antigravity.json",
    "frozen_predictions_antigravity.jsonl", "prediction_manifest_antigravity.json", "evaluation_antigravity.json",
    "candidate_metrics_antigravity.jsonl", "family_metrics_antigravity.jsonl", "bootstrap_antigravity.json",
    "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json",
)

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
    "experiment": "Exp454",
    "schema": "exp454",
    "kind": "exp453_exp449_authority_boundary_repair",
    "run_id": "exp454-exp453-exp449-authority-boundary-repair-antigravity-20260814",
    "fixed_interpreter": FIXED_INTERPRETER_ANTIGRAVITY,
    "fixed_executable": FIXED_EXECUTABLE_ANTIGRAVITY,
    "fixed_prefix": FIXED_PREFIX_ANTIGRAVITY,
    "runtime_versions": RUNTIME_VERSIONS_ANTIGRAVITY,
    "evaluation_leaves": EVALUATION_LEAVES_ANTIGRAVITY,
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
    "evaluation_root": EXP454_EVALUATION_ROOT_ANTIGRAVITY,
    "receipt_directory": EXP454_RECEIPT_DIRECTORY_ANTIGRAVITY,
    "failure_audit": EXP454_FAILURE_AUDIT_ANTIGRAVITY,
    "target_dataset_count": DATASET_COUNT_ANTIGRAVITY,
    "target_prediction_count": PREDICTION_COUNT_ANTIGRAVITY,
    "target_family_method_pair_count": FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY,
    "index_order_rule": INDEX_ORDER_RULE_ANTIGRAVITY,
})
