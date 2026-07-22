from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP328_ROOT = DATA_ROOT / "exp328_large_fn_reachability_audit"
EXP329_ROOT = DATA_ROOT / "exp329_large_candidate_train_verifier"
EXP330_ROOT = DATA_ROOT / "exp330_large_learned_normal_representation"
EXP329_ID = "experiment_329_large_candidate_train_verifier"
EXP330_ID = "experiment_330_large_learned_normal_representation"
EXPECTED_DATASETS = 251
ALPHAS = (0.005, 0.01, 0.02)
PCA_DIMENSIONS = (32, 64)
RANDOM_STATE = 20260723

