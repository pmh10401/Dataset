from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp308_frozen_six_expert_review_tiers"
EXPERIMENT_ID = "experiment_308_frozen_six_expert_review_tiers"
EXPECTED_DATASETS = 1117
EXPECTED_STANDARD = 806
GATE_VARIANT = "risk_first_distinct3"
STRONG_COUNT = 17
MEDIUM_COUNT = 57

EXP307_ROOT = DATA_ROOT / "exp307_expanded_six_expert_moe"
EXP307_PREDICTION = EXP307_ROOT / "exp307_frozen_predictions.jsonl"
EXP307_MANIFEST = EXP307_ROOT / "exp307_prediction_manifest.json"
EXP307_PREDICTION_SHA256 = "7f00ee9959a728f709fcf63ec6c8ae1cbd3e077db3b3e5911d7d0cdb94d46ccd"
