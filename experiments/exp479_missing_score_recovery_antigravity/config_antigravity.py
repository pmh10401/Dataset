"""Frozen result-affecting constants for Exp479."""

from __future__ import annotations

from pathlib import Path


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[2]
NUMBA_CACHE_DIR_ANTIGRAVITY = Path("/private/tmp/exp479_numba_cache_antigravity")
DATABASE_PATH_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
DATABASE_EXPECTED_SHA256_ANTIGRAVITY = (
    "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
)
EXP478_RESULT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp478_full_1117_official_routing_antigravity/results_antigravity.json"
)
SCORE_CHECKPOINT_DIR_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY
    / "experiments/exp_routing_label_free_antigravity/checkpoints"
)
EXPERIMENT_DIR_ANTIGRAVITY = Path(__file__).resolve().parent
SMOKE_CHECKPOINT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "smoke_checkpoint_antigravity.json"
)
SMOKE_RESULT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "smoke_results_antigravity.json"
)
OFFICIAL_CHECKPOINT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "checkpoint_antigravity.json"
)
OFFICIAL_RESULT_PATH_ANTIGRAVITY = (
    EXPERIMENT_DIR_ANTIGRAVITY / "results_antigravity.json"
)
DASHBOARD_PATH_ANTIGRAVITY = ROOT_ANTIGRAVITY / "docs/EXP479_DASHBOARD_antigravity.html"
IMPLEMENTATION_REPORT_PATH_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "docs/EXP479_IMPLEMENTATION_REPORT_antigravity.html"
)

EXPECTED_INCLUDED_DATASETS_ANTIGRAVITY = 1117
EXPECTED_EXISTING_SCORE_DATASETS_ANTIGRAVITY = 962
EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY = 155
EXPECTED_SCORE_CHECKPOINTS_ANTIGRAVITY = 23
EXCLUDED_DATASETS_ANTIGRAVITY = (
    "CornellWhaleChallenge",
    "Phoneme_normal_39",
)
METHODS_ANTIGRAVITY = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
)
SEED_ANTIGRAVITY = 20260721
ALPHA_ANTIGRAVITY = 0.005


__all__ = [name for name in globals() if name.endswith("_ANTIGRAVITY")]
