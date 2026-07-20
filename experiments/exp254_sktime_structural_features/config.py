from pathlib import Path

EXPERIMENT = 254
SOURCE = "sktime_structural_features"
EXPERIMENT_ID = f"experiment_{EXPERIMENT}_{SOURCE}"
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SEED = 20260723
MINIMUM_TRAIN = 5
CROSSFIT_FOLDS = 5
KNN_NEIGHBORS = 3
MAX_WORKING_LENGTH = 256
MIN_WORKING_LENGTH = 32
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp254_sktime_structural_features")
VENV_PYTHON = Path("/Users/minho/Documents/Dataset/venvs/tsai-sktime-py314/bin/python")
HEARTBEAT_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_dashboard_heartbeat.json")
EXTERNAL_STATUS_PATH = Path("/Users/minho/Documents/Dataset/rank_dashboard_external_live_run.json")
EXP253_PATH = Path("/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/frozen_predictions.jsonl")
