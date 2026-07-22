from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceSpec:
    """Frozen description of one prediction artifact consumed by Exp263."""

    experiment: str
    artifact_path: Path
    variant: str
    alpha: float | None
    scope: str


ALPHA = 0.005
MIN_TRAIN_FOR_PROMOTION = 100
REQUIRED_PROMOTION_AGREEMENT = 3
PRIORITY_AGREEMENT = 2
STAT_VARIANT = "combined_40"
STRUCTURAL_VARIANT = "structural_combined_440"
CHANGE_VARIANT = "change_point_combined_33"
MASK_VARIANT = "mask_025__seed_intersection"

DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
OUTPUT_ROOT = Path(
    "/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration"
)

SOURCE_CONTRACT: dict[str, SourceSpec] = {
    "baseline": SourceSpec(
        experiment="exp253",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/frozen_predictions.jsonl"
        ),
        variant="final_lanes",
        alpha=None,
        scope="all_1117",
    ),
    "statistical": SourceSpec(
        experiment="exp252",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/frozen_predictions.jsonl"
        ),
        variant=STAT_VARIANT,
        alpha=ALPHA,
        scope="all_1117",
    ),
    "structural": SourceSpec(
        experiment="exp254",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp254_sktime_structural_features/frozen_predictions.jsonl"
        ),
        variant=STRUCTURAL_VARIANT,
        alpha=ALPHA,
        scope="all_1117",
    ),
    "change_point": SourceSpec(
        experiment="exp255",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp255_change_point_features/frozen_predictions.jsonl"
        ),
        variant=CHANGE_VARIANT,
        alpha=ALPHA,
        scope="all_1117",
    ),
    "reconstruction": SourceSpec(
        experiment="exp257",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/frozen_predictions.jsonl"
        ),
        variant=MASK_VARIANT,
        alpha=ALPHA,
        scope="n_train_at_least_100",
    ),
    "missing_override": SourceSpec(
        experiment="exp260_262",
        artifact_path=Path(
            "/Users/minho/Documents/Dataset/exp260_262_missing_aware_alert/frozen_predictions.jsonl"
        ),
        variant="corrected_four_sources",
        alpha=ALPHA,
        scope="missing_20",
    ),
}
