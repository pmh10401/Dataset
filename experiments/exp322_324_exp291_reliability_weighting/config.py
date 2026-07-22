from pathlib import Path


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
EXP116_RESULTS = DATA_ROOT / "experiment_116_train_only_reliability_reranker_results.csv"
EXP291_ROOT = DATA_ROOT / "exp291_final_operational_routing_freeze"
OUTPUT_ROOTS = {
    322: DATA_ROOT / "exp322_exp291_weight_lineage_audit",
    323: DATA_ROOT / "exp323_exp116_reliability_selection_replay",
    324: DATA_ROOT / "exp324_exp291_reliability_route_parity",
}
EXPERIMENT_IDS = {
    322: "experiment_322_exp291_weight_lineage_audit",
    323: "experiment_323_exp116_reliability_selection_replay",
    324: "experiment_324_exp291_reliability_route_parity",
}
