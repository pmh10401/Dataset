from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp292_standard_review_evidence_tiering"
EXPERIMENT_ID = "experiment_292_standard_review_evidence_tiering"
ALPHA = 0.005
EXPECTED_DATASETS = 1117

EXP291_PATH = DATA_ROOT / "exp291_final_operational_routing_freeze" / "exp291_frozen_operational_routing.jsonl"
EXP291_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    sha256: str
    variants: tuple[str, ...]
    expected_datasets: int


SOURCE_SPECS = (
    SourceSpec(
        "statistical",
        DATA_ROOT / "exp252_sktime_statistical_features_safe_hurst" / "frozen_predictions.jsonl",
        "dcc6bf54bc1726716a239ccc8f29c87abb32f8a8c56245374799332603dbdae5",
        ("catch22_22", "combined_40", "hurst_3", "signature_6", "summary_9"),
        1117,
    ),
    SourceSpec(
        "structural",
        DATA_ROOT / "exp254_sktime_structural_features" / "frozen_predictions.jsonl",
        "fd796d0fcba5a0413790b0a17bea263fd9fb3ca5273e5c88a8746d476ae4d997",
        ("matrix_profile_16", "random_interval_224", "sax_transition_72", "sfa_hash_128", "structural_combined_440"),
        1117,
    ),
    SourceSpec(
        "change_point",
        DATA_ROOT / "exp255_change_point_features" / "frozen_predictions.jsonl",
        "de964382304df87a52bf99e589ddfd8df8c63eebe832fe92ec4a22b60cbdec4e",
        ("change_point_combined_33", "cusum_mosum_18", "pelt_l1_l2_10", "pelt_rbf_5"),
        1117,
    ),
    SourceSpec(
        "masked_reconstruction",
        DATA_ROOT / "exp257_mask_ratio_timeseries_sweep" / "frozen_predictions.jsonl",
        "965ea8200ba49d2e10b3240566986bbc83084a2ab96b6ba7a3a5122799cafdd0",
        (
            "mask_00625__default", "mask_00625__seed_intersection",
            "mask_0125__default", "mask_0125__seed_intersection",
            "mask_025__default", "mask_025__seed_intersection",
            "mask_multiscale_00625_0125__default", "mask_multiscale_00625_0125__seed_intersection",
        ),
        251,
    ),
    SourceSpec(
        "official_multirocket",
        DATA_ROOT / "exp249_250_official_mini_multirocket" / "experiment_250_official_multirocket" / "frozen_predictions.jsonl",
        "b07ef8a3093446701229e82f2794f47da81bbfb7f855f6e9b36650d112cd0800",
        ("kernels_1024_seed_intersection", "kernels_1024_seed_median", "kernels_512_seed_intersection", "kernels_512_seed_median"),
        1117,
    ),
)

