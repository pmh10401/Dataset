from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp304_reconstruction_fifth_source_common_support"
EXPERIMENT_ID = "experiment_304_reconstruction_fifth_source_common_support"
EXPECTED_DATASETS = 251
ALPHA = 0.005

EXP257_ROOT = DATA_ROOT / "exp257_mask_ratio_timeseries_sweep"
EXP257_AUDIT_PATH = EXP257_ROOT / "audit.json"
EXP257_AUDIT_SHA256 = "fbe71e4eb7a279fbc406845404c5d6ed52617d37285841c1aa59f431de241dcc"
EXP257_PREDICTION_PATH = EXP257_ROOT / "frozen_predictions.jsonl"
EXP257_PREDICTION_SHA256 = "965ea8200ba49d2e10b3240566986bbc83084a2ab96b6ba7a3a5122799cafdd0"

EXP294_ROOT = DATA_ROOT / "exp294_frozen_sparse_moe_review_router"
EXP294_PREDICTION_PATH = EXP294_ROOT / "exp294_frozen_predictions.jsonl"
EXP294_MANIFEST_PATH = EXP294_ROOT / "exp294_prediction_manifest.json"
EXP294_PREDICTION_SHA256 = "183f731af9a19059a4e2c18b9452ddde0c22e7bc898c2a017eb22f2aead1c048"
