from __future__ import annotations

from pathlib import Path


PRIMARY_ALPHA = 0.005
MIN_TRAIN_FOR_EXPANSION = 100
FORECAST_GROUP = "rff_forecast+tcn_seed_intersection+tcn_seed_median"

DATA_DIR = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_DIR / "exp274_275_large_fn_recovery"
EXP272_ROOT = DATA_DIR / "exp267_273_zero_support_fn_recovery"
EXP239_ROOT = DATA_DIR / "exp239_248_diverse_score_sources" / "experiment_239_multiscale_wavelet_energy_deviation"
EXP254_ROOT = DATA_DIR / "exp254_sktime_structural_features"

