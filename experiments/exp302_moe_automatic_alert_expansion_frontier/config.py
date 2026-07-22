from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp302_moe_automatic_alert_expansion_frontier"
EXPERIMENT_ID = "experiment_302_moe_automatic_alert_expansion_frontier"
EXPECTED_DATASETS = 1117
EXPECTED_HARD_ALERTS = 2218
EXPECTED_STANDARD_CANDIDATES = 806

EXP294_ROOT = DATA_ROOT / "exp294_frozen_sparse_moe_review_router"
EXP294_PREDICTION_PATH = EXP294_ROOT / "exp294_frozen_predictions.jsonl"
EXP294_MANIFEST_PATH = EXP294_ROOT / "exp294_prediction_manifest.json"
EXP294_PREDICTION_SHA256 = "183f731af9a19059a4e2c18b9452ddde0c22e7bc898c2a017eb22f2aead1c048"
