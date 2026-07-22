from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp303_expanded_train_only_gates"
EXPERIMENT_ID = "experiment_303_expanded_train_only_gates"
EXPECTED_DATASETS = 1117
EXPECTED_HARD_ALERTS = 2218
EXPECTED_STANDARD_CANDIDATES = 806

EXP293_AUDIT_PATH = DATA_ROOT / "exp293_train_only_moe_expert_audit" / "exp293_expert_audit.csv"
EXP293_AUDIT_SHA256 = "3fec7c95b3091c82d19b691aa821b57270d2c18b1cc1c1db4eb3cb8c859a9b7d"
EXP294_ROOT = DATA_ROOT / "exp294_frozen_sparse_moe_review_router"
EXP294_PREDICTION_PATH = EXP294_ROOT / "exp294_frozen_predictions.jsonl"
EXP294_MANIFEST_PATH = EXP294_ROOT / "exp294_prediction_manifest.json"
EXP294_PREDICTION_SHA256 = "183f731af9a19059a4e2c18b9452ddde0c22e7bc898c2a017eb22f2aead1c048"
