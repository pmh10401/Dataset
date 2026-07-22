from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp306_componentized_operational_routing"
EXPERIMENT_ID = "experiment_306_componentized_operational_routing"
EXPECTED_DATASETS = 1117

INPUTS = {
    "exp236": (
        DATA_ROOT / "exp236_detection_coverage_expansion" / "exp236_frozen_predictions.jsonl",
        "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524",
    ),
    "exp253": (
        DATA_ROOT / "exp253_guarded_standard_to_hard_expansion" / "frozen_predictions.jsonl",
        "1ed2046a3aa934fd57675a478a2571f153f819849af30b71713216cd188570a4",
    ),
    "exp290": (
        DATA_ROOT / "exp290_train_raw_exact_normal_guard" / "exp290_frozen_predictions.jsonl",
        "4fb4f2be62000fc4f9c8110a2e61b56996c99132120af6160dd56a20779483ce",
    ),
    "exp291": (
        DATA_ROOT / "exp291_final_operational_routing_freeze" / "exp291_frozen_operational_routing.jsonl",
        "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
    ),
}
