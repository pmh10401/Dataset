from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp295_moe_hard_alert_confidence_tiering"
EXPERIMENT_ID = "experiment_295_moe_hard_alert_confidence_tiering"
EXPECTED_DATASETS = 1117
EXPECTED_HARD_ALERTS = 2218

EXP291_PATH = DATA_ROOT / "exp291_final_operational_routing_freeze" / "exp291_frozen_operational_routing.jsonl"
EXP291_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"
EXP293_ROOT = DATA_ROOT / "exp293_train_only_moe_expert_audit"
