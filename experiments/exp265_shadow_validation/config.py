from __future__ import annotations

from pathlib import Path


SCHEMA_VERSION = 1
POLICY_VERSION = "exp265-shadow-v1"
COLLECTION_MODE = "prospective_shadow"
ALPHA = 0.005
MIN_TRAIN_FOR_SHADOW = 100
SHADOW_AGREEMENT = 3
PRIORITY_AGREEMENT = 2
TARGET_NEW_RUNS = 100
SOURCE_NAMES = ("statistical", "structural", "change_point", "reconstruction")
FORBIDDEN_FIELDS = frozenset(
    {
        "y_test",
        "labels",
        "actual_anomaly_positions",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "combined_f1",
    }
)
OUTPUT_ROOT = Path("/Users/minho/Documents/Dataset/exp265_shadow_validation")
SPOOL_PATH = OUTPUT_ROOT / "exp265_spool.sqlite"

