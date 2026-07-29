"""Portable filesystem paths for the frozen Exp403-408 workflow."""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT_ENV = "TIMESERIES_DATA_ROOT"
HISTORICAL_DATA_ROOT = Path("/Users/minho/Documents/Dataset")


def resolve_data_root() -> Path:
    configured = os.environ.get(DATA_ROOT_ENV)
    if configured is None:
        return HISTORICAL_DATA_ROOT
    return Path(os.path.expandvars(configured)).expanduser().resolve(strict=False)


DATA_ROOT = resolve_data_root()
DB_PATH = DATA_ROOT / "univariate_ts.db"
EXP291_PREDICTION_PATH = (
    DATA_ROOT
    / "exp291_final_operational_routing_freeze"
    / "exp291_frozen_operational_routing.jsonl"
)
EXP291_MANIFEST_PATH = EXP291_PREDICTION_PATH.with_name(
    "exp291_prediction_manifest.json"
)
EXP292_PREDICTION_PATH = (
    DATA_ROOT
    / "exp292_standard_review_evidence_tiering"
    / "exp292_frozen_review_tiers.jsonl"
)
EXP292_MANIFEST_PATH = EXP292_PREDICTION_PATH.with_name(
    "exp292_prediction_manifest.json"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "exp403_408_feature_ml_validation"
EXP417_BACKEND_AUDIT_PATH = DATA_ROOT / "exp417_causal_multihorizon_shadow" / "task7_backend_audit.json"
