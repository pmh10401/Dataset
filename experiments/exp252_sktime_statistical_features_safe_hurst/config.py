from __future__ import annotations

from pathlib import Path


EXPERIMENT = 252
SOURCE = "sktime_statistical_features_safe_hurst"
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SEED = 20260722
MINIMUM_TRAIN = 5
CROSSFIT_FOLDS = 5
KNN_NEIGHBORS = 3
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst")
VENV_PYTHON = Path("/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python")
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")
EXTERNAL_STATUS_PATH = Path("/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json")

