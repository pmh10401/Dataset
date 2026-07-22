from __future__ import annotations


MIN_CALIBRATION_ROWS = 5
ELIGIBLE_TRAIN_ROWS = 50
MAX_ATTAINABLE_P = 0.02
MAX_RISK_UPPER = 0.005
MIN_SOURCE_STABILITY = 0.75
MAX_TRAIN_DRIFT = 0.25


def reliability_state(
    *,
    n_train: int,
    attainable_p: float,
    risk_upper: float,
    source_stability: float,
    train_drift: float,
) -> str:
    if int(n_train) < MIN_CALIBRATION_ROWS:
        return "abstain"
    eligible = (
        int(n_train) >= ELIGIBLE_TRAIN_ROWS
        and float(attainable_p) <= MAX_ATTAINABLE_P
        and float(risk_upper) <= MAX_RISK_UPPER
        and float(source_stability) >= MIN_SOURCE_STABILITY
        and float(train_drift) <= MAX_TRAIN_DRIFT
    )
    return "eligible" if eligible else "review_only"

