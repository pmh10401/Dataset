from __future__ import annotations

from pathlib import Path

from experiments.exp317_318_train_reliability_fusion.config import ALPHAS, DATA_ROOT, SOURCE_GROUPS


CONFIDENCE_THRESHOLDS = (0.50, 0.67, 0.75)
TOP_KS = (20, 50, 100)
SEED = 20260722
SPLITS = 5
EXP317_ROOT = DATA_ROOT / "exp317_train_normal_source_reliability_audit"
OUTPUT_ROOTS = {
    319: DATA_ROOT / "exp319_candidate_level_train_reliability",
    320: DATA_ROOT / "exp320_existing_review_reliability_ranking",
    321: DATA_ROOT / "exp321_frozen_dual_lane_architecture",
}
EXPERIMENT_IDS = {
    319: "experiment_319_candidate_level_train_reliability",
    320: "experiment_320_existing_review_reliability_ranking",
    321: "experiment_321_frozen_dual_lane_architecture",
}
