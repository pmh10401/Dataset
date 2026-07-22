from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
ALPHA = 0.005
EXPECTED_DATASETS = 1117
EXP293_ID = "experiment_293_train_only_moe_expert_audit"
EXP294_ID = "experiment_294_frozen_sparse_moe_review_router"
EXP293_ROOT = DATA_ROOT / "exp293_train_only_moe_expert_audit"
EXP294_ROOT = DATA_ROOT / "exp294_frozen_sparse_moe_review_router"
EXP292_PATH = DATA_ROOT / "exp292_standard_review_evidence_tiering" / "exp292_frozen_review_tiers.jsonl"
EXP292_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
WORKLOADS = (17, 74, 806)


@dataclass(frozen=True)
class CacheSpec:
    expert: str
    root: Path
    manifest: Path
    manifest_sha256: str
    config_hash: str
    train_keys: tuple[str, ...]


CACHE_SPECS = (
    CacheSpec(
        "statistical",
        DATA_ROOT / "exp252_sktime_statistical_features_safe_hurst",
        DATA_ROOT / "exp252_sktime_statistical_features_safe_hurst" / "cache_manifest.csv",
        "883d6026ce92c27282aaec67f6dd1f4632fe202052a3afdb9c9646d257a89f5a",
        "28d118cdc83f9ffec97db7bc569316ee59237e120291306bb3144d5562bfc227",
        ("catch22_22__train_oof", "summary_9__train_oof", "signature_6__train_oof", "hurst_3__train_oof", "combined_40__train_oof"),
    ),
    CacheSpec(
        "structural",
        DATA_ROOT / "exp254_sktime_structural_features",
        DATA_ROOT / "exp254_sktime_structural_features" / "cache_manifest.csv",
        "1a3e68309364dc1b82e39456f5d4a2ad5508d3d2e0fa8c347d6569316f419b98",
        "0f357e518bde16a7735cf6a2951b9ef87950a0823d72122f9744aaab191fb363",
        ("random_interval_224__train_oof", "matrix_profile_16__train_oof", "sax_transition_72__train_oof", "sfa_hash_128__train_oof", "structural_combined_440__train_oof"),
    ),
    CacheSpec(
        "change_point",
        DATA_ROOT / "exp255_change_point_features",
        DATA_ROOT / "exp255_change_point_features" / "cache_manifest.csv",
        "6cffdd39920ba807262f2ca469b9efb42f1c55d4b7b4ed4602f1420a509d393d",
        "b4b37152744ef6cb3631ba6638fc6f7e1912a0ddef28610e5fbd0480e0c04efc",
        ("cusum_mosum_18__train_oof", "pelt_l1_l2_10__train_oof", "pelt_rbf_5__train_oof", "change_point_combined_33__train_oof"),
    ),
    CacheSpec(
        "official_multirocket",
        DATA_ROOT / "exp249_250_official_mini_multirocket" / "experiment_250_official_multirocket",
        DATA_ROOT / "exp249_250_official_mini_multirocket" / "experiment_250_official_multirocket" / "cache_manifest.csv",
        "742177c573d91bee1de461a86c3727483fc99bb9dd74e52cad071c5c5d6429fd",
        "ef36cdcc1763ace04d8a29643bb5ea301f284c66b0b63c697d66c8ee7f1ab53a",
        ("kernels_512__train_by_seed", "kernels_1024__train_by_seed"),
    ),
)

RECON_ROOT = DATA_ROOT / "exp257_mask_ratio_timeseries_sweep" / "cache"
RECON_INVENTORY = Path(__file__).resolve().parents[1] / "exp266_fn_recovery_shadow" / "reconstruction_archive_hash_inventory.json"
RECON_INVENTORY_SHA256 = "7efe6c80346175bed8d88d340db50524c9fffbd30794896db77809580117745f"
RECON_KEYS = (
    "mask_00625__train_by_seed", "mask_0125__train_by_seed",
    "mask_025__train_by_seed", "mask_multiscale_00625_0125__train_by_seed",
)

