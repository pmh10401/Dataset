from __future__ import annotations

from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SHARE_THRESHOLDS = (0.50, 0.67, 0.75)
SEED = 20260722
RELIABILITY_SPLITS = 5
MIN_TRAIN_SCORES = 20
WILSON_Z = 1.6448536269514722
EXPECTED_DATASETS = 1117

EXP291_ROOT = DATA_ROOT / "exp291_final_operational_routing_freeze"
EXP291_PREDICTION = EXP291_ROOT / "exp291_frozen_operational_routing.jsonl"
EXP291_MANIFEST = EXP291_ROOT / "exp291_prediction_manifest.json"
EXP290_ROOT = DATA_ROOT / "exp290_train_raw_exact_normal_guard"

SOURCE_EXPERIMENTS = {
    "multiscale_patch_discord": 309,
    "multiwindow_change_point": 310,
    "wavelet_phase_coherence": 311,
    "predictive_residual": 313,
    "contrastive_normal_memory": 314,
    "deep_svdd_normal_boundary": 315,
}

SOURCE_GROUPS = {
    "multiscale_patch_discord": "local_shape",
    "multiwindow_change_point": "transition",
    "wavelet_phase_coherence": "frequency",
    "predictive_residual": "prediction",
    "contrastive_normal_memory": "representation",
    "deep_svdd_normal_boundary": "representation",
}

SOURCE_ROOTS = {
    source: DATA_ROOT / f"exp{number}_{source}"
    for source, number in SOURCE_EXPERIMENTS.items()
}

EXPERIMENT_IDS = {
    317: "experiment_317_train_normal_source_reliability_audit",
    318: "experiment_318_train_reliability_weighted_fusion",
}

OUTPUT_ROOTS = {
    number: DATA_ROOT / experiment_id.replace("experiment_", "exp", 1)
    for number, experiment_id in EXPERIMENT_IDS.items()
}
