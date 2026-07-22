from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp307_expanded_six_expert_moe"
EXPERIMENT_ID = "experiment_307_expanded_six_expert_moe"
EXPECTED_DATASETS = 1117
EXPECTED_STANDARD = 806
ALPHA = 0.005
WORKLOADS = (17, 74, 806)

EXP292_PATH = DATA_ROOT / "exp292_standard_review_evidence_tiering" / "exp292_frozen_review_tiers.jsonl"
EXP292_SHA256 = "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b"
EXP293_AUDIT = DATA_ROOT / "exp293_train_only_moe_expert_audit" / "exp293_expert_audit.csv"
EXP293_AUDIT_SHA256 = "3fec7c95b3091c82d19b691aa821b57270d2c18b1cc1c1db4eb3cb8c859a9b7d"
EXP306_ROOT = DATA_ROOT / "exp306_componentized_operational_routing"

SOURCE_SPECS = {
    "ordinal_pattern": {
        "root": DATA_ROOT / "exp239_248_diverse_score_sources" / "experiment_240_ordinal_pattern_distribution_shift",
        "cache_manifest_sha256": "2d2db413e8a03d04338a660feae5e18be1906fd7fe14939d67c65761daae479b",
        "prediction_sha256": "82de3fcb4dfd5cbecd4a521f5f2fc0d2f869ba32c14a9151a0c142e3ebe90430",
        "variant": "default",
    },
    "masked_context": {
        "root": DATA_ROOT / "exp239_248_diverse_score_sources" / "experiment_247_masked_patch_context_error",
        "cache_manifest_sha256": "ebd25fd74d1ed94523bc15bfcbc1dae34994a89e71b085c1a19264cef6618266",
        "prediction_sha256": "ef4afc7d92913e90b2a87ad3c7b5e79a282c0db011c3796663754a85856670ff",
        "variant": "seed_intersection",
    },
}
