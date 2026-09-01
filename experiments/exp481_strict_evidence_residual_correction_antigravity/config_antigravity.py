"""Frozen, result-affecting Exp481 configuration."""

from __future__ import annotations

from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
EXP480_PACKAGE_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp480_family_balanced_evidence_routing_antigravity"
)
DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
OFFICIAL_ROW_KNN_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py"
)
EXP480_EVIDENCE_DATABASE_PATH_ANTIGRAVITY = (
    EXP480_PACKAGE_ANTIGRAVITY / "evidence_antigravity.sqlite3"
)
EXP480_RESULT_PATH_ANTIGRAVITY = (
    EXP480_PACKAGE_ANTIGRAVITY / "results_antigravity.json"
)
SMOKE_RESULT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "smoke_results_antigravity.json"
)
OFFICIAL_RESULT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
)
RUN_MANIFEST_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "run_manifest_antigravity.json"
)
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP481_DASHBOARD_antigravity.html"
IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "docs/EXP481_IMPLEMENTATION_REPORT_antigravity.html"
)
ACCELERATION_ASSESSMENT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/exp481_acceleration_assessment_antigravity.json"
)

EXPECTED_DATASETS_ANTIGRAVITY = 1117
EXPECTED_FAMILIES_ANTIGRAVITY = 128
EXPECTED_EXP480_EVIDENCE_SHA256_ANTIGRAVITY = (
    "77c53e1afd0af071edf71ef692bd978458dda6a5f8a0fec110736b787fe1b998"
)
EXPECTED_EXP480_RESULT_SHA256_ANTIGRAVITY = (
    "bfe65c92aa5890bcffa90d1febe4c2d902219a66eb5d7e50361a29e94458d566"
)
METHODS_ANTIGRAVITY = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
)
FEATURE_KEYS_ANTIGRAVITY = (
    "width",
    "n_train",
    "spectral_entropy",
    "skewness",
    "kurtosis",
    "diff_std_ratio",
    "zero_crossing_rate",
)
BASELINE_METHOD_ANTIGRAVITY = METHODS_ANTIGRAVITY[0]
ALTERNATIVE_METHODS_ANTIGRAVITY = METHODS_ANTIGRAVITY[1:]
K_ANTIGRAVITY = 31
INNER_MIN_CHANGED_FAMILIES_ANTIGRAVITY = 20
AUC_QUANTILE_ANTIGRAVITY = 0.20


__all__ = [name for name in globals() if name.endswith("_ANTIGRAVITY")]
