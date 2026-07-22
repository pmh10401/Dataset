from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp297_bootstrap_confidence_stability"
EXPERIMENT_ID = "experiment_297_bootstrap_confidence_stability"
EXPECTED_DATASETS = 1117
EXPECTED_HARD_ALERTS = 2218
BOOTSTRAP_REPLICATES = 32
BOOTSTRAP_SEED = 20260722
ALPHA = 0.005

EXP295_ROOT = DATA_ROOT / "exp295_moe_hard_alert_confidence_tiering"
EXP295_PREDICTION_SHA256 = "59c76a97bc924109242bbe3f13089c84010091271a6be33ddcb3bb78de3037b0"
EXP296_ROOT = DATA_ROOT / "exp296_cross_gate_hard_confidence_consensus"
EXP296_PREDICTION_SHA256 = "5b321d207222772cbbbe95ee12ecc8cee20fe281e2e3a7c00eea58eb067c7178"
