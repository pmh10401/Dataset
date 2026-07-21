from __future__ import annotations

from pathlib import Path


EXPERIMENT = 266
EXPERIMENT_ID = "experiment_266_outside_lane_fn_recovery_shadow"
ALPHA = 0.005
SEED = 20260721
EXPECTED_DATASETS = 1117
RISK_TARGET = 0.01
MINIMUM_TRAIN = 5
FULL_RUN_WORKERS = 7

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
RECON_ARCHIVE_HASH_INVENTORY = Path(__file__).with_name(
    "reconstruction_archive_hash_inventory.json"
)
FROZEN_RECON_ARCHIVE_HASH_INVENTORY_SHA256 = (
    "7efe6c80346175bed8d88d340db50524c9fffbd30794896db77809580117745f"
)

FROZEN_DATABASE_EXCLUSIONS = frozenset({"CornellWhaleChallenge", "Wafer_normal_1"})
FROZEN_PREDICTION_HASHES = {
    "exp253": "1ed2046a3aa934fd57675a478a2571f153f819849af30b71713216cd188570a4",
    "exp264": "8c7a2f52027c6a3127c79166bb109b61e348558574d98a42a8ccf5622921d653",
    "minirocket": "fc72d2dfbaf6a1653ec804dbe349397214992387083b22bdab0e8c26a66876c2",
}
FROZEN_PREDICTION_MANIFEST_HASHES = {
    "exp253": "4886b67d294dc87c2a8c22f4616d72860089bd26f140b4ae7cee1108a2267692",
    "exp264": "c01800fa8729bcba08d73a8ec407b7284297919be9b008bb8b0f36e397d53387",
    "minirocket": "25ade7c3f8362559a29c15e43fead6070a7fac6cab648f03f62bcb20382b8c8f",
}
FROZEN_PREDICTION_AUDIT_HASHES = {
    "exp253": "bd16898761061c918a9bc9581494949fb7d0e424afc4bc602137cc123f281f19",
    "exp264": "41fad73740be8bc75f722a258f77cdfed3b2a77bed4f19a201f74d558611e492",
    "minirocket": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
}
FROZEN_AUDIT_HASHES = {
    "kernel": "144ae3f65fea3b4a8cf306fd54b097788c64666dc2d21b673fbf65500c12e0ee",
    "statistical": "a10ed60e138d289de6c659d31f19cd6ad41cd73064b28e7aa50ec007c15ce507",
    "structural": "3bd3556b3bc7d2216796218a60c31f9fab4edadfc7371e003dfb972546bf738b",
    "change_point": "36037886f957033bf371dd52c187edbdb34d645b87dffe710bf9164d51b15bc3",
    "reconstruction": "fbe71e4eb7a279fbc406845404c5d6ed52617d37285841c1aa59f431de241dcc",
}
FROZEN_CACHE_MANIFEST_HASHES = {
    "kernel": "44c0ecc80ff7cbae383d49d2b8b0cefed442c49b8a12a7bd0cf8569ce121c971",
    "statistical": "883d6026ce92c27282aaec67f6dd1f4632fe202052a3afdb9c9646d257a89f5a",
    "structural": "1a3e68309364dc1b82e39456f5d4a2ad5508d3d2e0fa8c347d6569316f419b98",
    "change_point": "6cffdd39920ba807262f2ca469b9efb42f1c55d4b7b4ed4602f1420a509d393d",
}
FROZEN_AUDIT_CONFIGS = {
    "kernel": {
        "experiment": 249,
        "source": "official_minirocket",
        "seeds": [20260721, 20261721],
        "train_calibration": "five_fold_transformer_crossfit",
        "requested_kernels": [1024, 4096],
    },
    "statistical": {
        "experiment": 252,
        "source": "sktime_statistical_features_safe_hurst",
        "seed": 20260722,
        "train_calibration": "five_fold_oof",
    },
    "structural": {
        "experiment": 254,
        "source": "sktime_structural_features",
        "seed": 20260723,
        "train_calibration": "five_fold_oof",
    },
    "change_point": {
        "experiment": 255,
        "source": "change_point_features",
        "seed": 20260724,
        "train_calibration": "five_fold_oof",
    },
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
