from __future__ import annotations

from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
SEED = 20260722
ALPHAS = (0.005, 0.01, 0.02, 0.05)
CANDIDATE_GRAIN = "test_instance_virtual_run"
CPU_WORKERS = 7
EXPECTED_DATASETS = 1117

EXPERIMENT_IDS = {
    309: "experiment_309_multiscale_patch_discord",
    310: "experiment_310_multiwindow_change_point",
    311: "experiment_311_wavelet_phase_coherence",
    312: "experiment_312_nonlearning_evidence_fusion",
    313: "experiment_313_predictive_residual",
    314: "experiment_314_contrastive_normal_memory",
    315: "experiment_315_deep_svdd_normal_boundary",
    316: "experiment_316_full_fn_recovery_fusion",
}

OUTPUT_ROOTS = {
    number: DATA_ROOT / experiment_id.replace("experiment_", "exp", 1)
    for number, experiment_id in EXPERIMENT_IDS.items()
}

EXP291_ROOT = DATA_ROOT / "exp291_final_operational_routing_freeze"
EXP291_PREDICTION = EXP291_ROOT / "exp291_frozen_operational_routing.jsonl"
EXP291_MANIFEST = EXP291_ROOT / "exp291_prediction_manifest.json"
EXP290_ROOT = DATA_ROOT / "exp290_train_raw_exact_normal_guard"

DISCORD_SCALE_FRACTIONS = (1 / 64, 1 / 32, 1 / 16, 1 / 8)
CHANGE_WINDOW_FRACTIONS = (1 / 32, 1 / 16, 1 / 8, 1 / 4)
WAVELET_SCALES = (2, 4, 8, 16, 32)
MAX_PATCH_MEMORY = 2048

NEURAL_MIN_TRAIN = 20
NEURAL_MIN_LENGTH = 32
NEURAL_EPOCHS = 30
NEURAL_PATIENCE = 5
NEURAL_BATCH_SIZE = 128
NEURAL_LEARNING_RATE = 1e-3
