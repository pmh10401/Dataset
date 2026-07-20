from pathlib import Path

EXPERIMENT = 253
SOURCE = "guarded_standard_to_hard_expansion"
EXPERIMENT_ID = f"experiment_{EXPERIMENT}_{SOURCE}"
ALPHA = 0.005
MINI_VARIANTS = {
    "mini1024": ("kernels_1024_seed_intersection", ALPHA),
    "mini4096": ("kernels_4096_seed_intersection", ALPHA),
}
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion")
EXP236_PATH = Path("/Users/minho/Documents/Dataset/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")
EXP249_PATH = Path("/Users/minho/Documents/Dataset/exp249_250_official_mini_multirocket/experiment_249_official_minirocket/frozen_predictions.jsonl")
EVIDENCE_SOURCES = {
    "ordinal": (Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/experiment_240_ordinal_pattern_distribution_shift/frozen_predictions.jsonl"), "default"),
    "complexity": (Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/experiment_242_complexity_signature_deviation/frozen_predictions.jsonl"), "default"),
    "fourier": (Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/experiment_245_smooth_fourier_normal_residual/frozen_predictions.jsonl"), "default"),
    "masked": (Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources/experiment_247_masked_patch_context_error/frozen_predictions.jsonl"), "seed_intersection"),
}
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")
EXTERNAL_STATUS_PATH = Path("/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json")

