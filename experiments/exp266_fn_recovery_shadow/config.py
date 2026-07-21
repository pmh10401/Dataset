from __future__ import annotations

from pathlib import Path


EXPERIMENT = 266
EXPERIMENT_ID = "experiment_266_outside_lane_fn_recovery_shadow"
ALPHA = 0.005
SEED = 20260721
EXPECTED_DATASETS = 1117
RISK_TARGET = 0.01
MINIMUM_TRAIN = 5

DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp266_outside_lane_fn_recovery_shadow")
EXP253_PREDICTION = Path(
    "/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/"
    "frozen_predictions.jsonl"
)
EXP264_PREDICTION = Path(
    "/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/"
    "frozen_predictions.jsonl"
)
MINIROCKET_PREDICTION = Path(
    "/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket/"
    "experiment_249_official_minirocket/frozen_predictions.jsonl"
)
MINI_CACHE_ROOT = MINIROCKET_PREDICTION.parent / "cache"
STAT_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/cache")
STRUCT_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp254_sktime_structural_features/cache")
CHANGE_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp255_change_point_features/cache")
RECON_CACHE_ROOT = Path("/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/cache")

FROZEN_DATABASE_EXCLUSIONS = frozenset({"CornellWhaleChallenge", "Wafer_normal_1"})
FROZEN_PREDICTION_HASHES = {
    "exp253": "1ed2046a3aa934fd57675a478a2571f153f819849af30b71713216cd188570a4",
    "exp264": "8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653",
    "minirocket": "fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2",
}
FROZEN_AUDIT_HASHES = {
    "kernel": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
    "statistical": "a10ed60e138d289de6c659d31f19cd6ad41cd73064b28e7aa50ec007c15ce507",
    "structural": "3bd3556b3bc7d2216796218a60c31f9fab4edadfc7371e003dfb972546bf738b",
    "change_point": "36037886f957033bf371dd52c187edbdb34d645b87dffe710bf9164d51b15bc3",
    "reconstruction": "fbe71e4eb7a279fbc406845404c5d6ed52617d37285841c1aa59f431de241dcc",
}
EXPECTED_LANE_TOTALS = {
    "hard": 2253,
    "shadow": 2,
    "priority": 31,
    "standard": 806,
    "union": 3092,
}
CACHE_CONTRACTS = (
    (
        "kernel",
        249,
        "official_minirocket",
        MINI_CACHE_ROOT,
        "8ae2ab74764db26d76e563030ad99346fa3c1a6ef97164c3b5d590b443735921",
        "five_fold_transformer_crossfit",
    ),
    (
        "statistical",
        252,
        "sktime_statistical_features_safe_hurst",
        STAT_CACHE_ROOT,
        "28d118cdc83f9ffec97db7bc569316ee59237e120291306bb3144d5562bfc227",
        "five_fold_oof",
    ),
    (
        "structural",
        254,
        "sktime_structural_features",
        STRUCT_CACHE_ROOT,
        "0f357e518bde16a7735cf6a2951b9ef87950a0823d72122f9744aaab191fb363",
        "five_fold_oof",
    ),
    (
        "change_point",
        255,
        "change_point_features",
        CHANGE_CACHE_ROOT,
        "b4b37152744ef6cb3631ba6638fc6f7e1912a0ddef28610e5fbd0480e0c04efc",
        "five_fold_oof",
    ),
)

MINI_VARIANTS = (
    "kernels_1024_seed_intersection",
    "kernels_4096_seed_intersection",
)
EVIDENCE_GROUPS = (
    "kernel",
    "statistical",
    "structural",
    "change_point",
    "reconstruction",
)
