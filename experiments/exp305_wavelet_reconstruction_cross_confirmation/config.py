from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp305_wavelet_reconstruction_cross_confirmation"
EXPERIMENT_ID = "experiment_305_wavelet_reconstruction_cross_confirmation"
EXPECTED_DATASETS = 251
ALPHA = 0.005

EXP304_ROOT = DATA_ROOT / "exp304_reconstruction_fifth_source_common_support"
EXP304_PREDICTION_PATH = EXP304_ROOT / "exp304_frozen_predictions.jsonl"
EXP304_MANIFEST_PATH = EXP304_ROOT / "exp304_prediction_manifest.json"
EXP304_PREDICTION_SHA256 = "c55b1f57b6551fca26037f01d61261e0a53d9b4ff6b5feedb753b9b0a0d70aa1"

EXP239_ROOT = (
    DATA_ROOT
    / "exp239_248_diverse_score_sources"
    / "experiment_239_multiscale_wavelet_energy_deviation"
)
EXP239_PREDICTION_PATH = EXP239_ROOT / "frozen_predictions.jsonl"
EXP239_MANIFEST_PATH = EXP239_ROOT / "prediction_manifest.json"
EXP239_PREDICTION_SHA256 = "4d8032980914843099b75f2ab39b22aba04404bdb814596cc4f8561a4951f0d2"
