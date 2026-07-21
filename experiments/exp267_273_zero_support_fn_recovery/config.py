from __future__ import annotations

from pathlib import Path

from .contracts import SourceSpec


SEED = 20260721
SECOND_SEED = 20261721
ALPHAS = (0.005, 0.01, 0.02, 0.05)
PRIMARY_ALPHA = 0.005
MIN_TRAIN_FOR_CALIBRATION = 5
CALIBRATION_MIN_TRAIN = MIN_TRAIN_FOR_CALIBRATION
CALIBRATION_ABSTENTION_REASON = "n_train_below_5"
RFF_MIN_TRAIN = 20
RFF_MAX_TRAIN_RESIDUAL_VALUES = 10_000_000
TCN_MIN_TRAIN = 100
REVIEW_RISK_LIMIT = 0.01
HARD_RISK_LIMIT = 0.005
MIN_RETROSPECTIVE_PRECISION = 0.80

DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery")

SOURCE_SPECS = (
    SourceSpec("scattering", "scattering", MIN_TRAIN_FOR_CALIBRATION, ("primary",)),
    SourceSpec("signature", "signature", MIN_TRAIN_FOR_CALIBRATION, ("primary",)),
    SourceSpec("rff_forecast", "normal_forecast", RFF_MIN_TRAIN, ("primary",)),
    SourceSpec(
        "tcn_forecast",
        "normal_forecast",
        TCN_MIN_TRAIN,
        ("seed_median", "seed_intersection"),
    ),
)
