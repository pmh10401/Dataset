from __future__ import annotations

from pathlib import Path


SEED = 20260719
ALPHAS = (0.005, 0.01, 0.02, 0.05)
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp237_new_score_source_qualification")
REPO_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "exp237_new_score_source_qualification"
)
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")

SOURCE_NAMES = (
    "pooled_ar_residual",
    "robust_change_statistics",
    "shrinkage_normal_space",
    "elastic_normal_prototype",
    "subsequence_discord",
)
SOURCE_MAX_LENGTH = 256
