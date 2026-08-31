"""Frozen, result-affecting Exp480 configuration."""

from __future__ import annotations

from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
CHECKPOINT_DIR_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "experiments/exp_routing_label_free_antigravity/checkpoints"
)
EXP478_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp478_full_1117_official_routing_antigravity/results_antigravity.json"
)
EXP479_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp479_missing_score_recovery_antigravity/results_antigravity.json"
)
FEATURE_CACHE_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp474_conservative_fallback_routing_antigravity/train_features_antigravity.json"
)
OFFICIAL_FEATURE_LOADER_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/official_router_antigravity.py"
)
EVIDENCE_DATABASE_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "evidence_antigravity.sqlite3"
)
SMOKE_RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "smoke_results_antigravity.json"
OFFICIAL_RESULT_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
RUN_MANIFEST_PATH_ANTIGRAVITY = EXPERIMENT_DIR_ANTIGRAVITY / "run_manifest_antigravity.json"
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP480_DASHBOARD_antigravity.html"
IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "docs/EXP480_IMPLEMENTATION_REPORT_antigravity.html"
)
MLX_ASSESSMENT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/exp480_mlx_assessment_antigravity.json"
)

EXPECTED_DATABASE_SHA256_ANTIGRAVITY = (
    "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
)
EXPECTED_EXP478_SHA256_ANTIGRAVITY = (
    "7af31a8849c00af9a3021fea60f9cedc4e0b537b0a1ac483d4f5c5f7ec0d26cc"
)
EXPECTED_EXP479_SHA256_ANTIGRAVITY = (
    "88680cf17487b87996aa729a7f5b3787e7d8e5013f0c672b32c26d49bf8d8110"
)
EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY = (
    "10cf37c2eca6336eacf25badb85653e75232dfaf7b6caeef66abb149241d20ba"
)
EXPECTED_FEATURE_LOADER_SHA256_ANTIGRAVITY = (
    "9d42dad06e92ae136a760db8771506377b3ccdca8615122ce1b0b43bca0b927c"
)

EXPECTED_DATASETS_ANTIGRAVITY = 1117
EXPECTED_FAMILIES_ANTIGRAVITY = 128
EXPECTED_FROZEN_962_ANTIGRAVITY = 962
EXPECTED_RECOVERED_155_ANTIGRAVITY = 155
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
FAMILY_POLICY_CONFIG_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
}
ROW_POLICY_CONFIG_ANTIGRAVITY = {
    "k": 31,
    "min_support": 20,
    "positive_rate_threshold": 0.90,
    "lower_quantile": 0.20,
    "min_auc_gain": 0.0,
    "min_zf_gain": 0.0,
    "min_lower_zf_gain": 0.0,
}


__all__ = [name for name in globals() if name.endswith("_ANTIGRAVITY")]
