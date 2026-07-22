from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP87_RESULTS = DATA_ROOT / "experiment_87_exp84_index_diagnostics_results.csv"
EXP90_RESULTS = DATA_ROOT / "experiment_90_zero_f1_repair_selector_results.csv"
EXP119A_RESULTS = DATA_ROOT / "experiment_119a_exp93_rank_order_validation_results.csv"
EXP291_ROOT = DATA_ROOT / "exp291_final_operational_routing_freeze"

EXP87_CONFIG = "aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3"
EXP90_SELECTOR = "noalert_top1_train_safe_repair"
EXP119A_SELECTOR = "exp93_rank_order_validated"
SEED = 20260722
MAX_TOP = 12
POOL_TOP = 8
EXPECTED_DATASETS = 1117

FIXED_PRIORS = {
    "rocket": 0.36,
    "frequency_image": 0.22,
    "texture_shape": 0.22,
    "aux_family_guard": 0.12,
    "aux_count_cap": 0.08,
}

EXPERIMENT_IDS = {
    325: "experiment_325_full_source_train_reliability_cache",
    326: "experiment_326_full_source_reliability_exp93_replay",
    327: "experiment_327_exp291_upstream_reliability_impact_audit",
}
OUTPUT_ROOTS = {
    number: DATA_ROOT / experiment_id.replace("experiment_", "exp", 1)
    for number, experiment_id in EXPERIMENT_IDS.items()
}
