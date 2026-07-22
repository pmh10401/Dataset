from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp249_250_official_mini_multirocket.cache_pipeline import cache_paths as multirocket_cache_paths
from experiments.exp254_sktime_structural_features.cache_pipeline import cache_paths as structural_cache_paths
from experiments.exp255_change_point_features.cache_pipeline import cache_paths as change_point_cache_paths
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp292_standard_review_evidence_tiering.contracts import freeze_prediction_rows, sha256_file, verify_prediction_rows

from .core import (
    AUTOMATIC_ALPHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPERIMENT_ID,
    GENERAL_ALPHA,
    PRIORITY_ALPHA,
    TRAIN_RISK_RATE_MAXIMUM,
    TRAIN_RISK_UPPER_MAXIMUM,
    bootstrap_consensus_stability,
    build_routing_row,
    consensus_train_flags,
    evaluate_predictions,
    summarize_evaluated,
    wilson_upper,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
OUTPUT_ROOT = DATA_ROOT / "exp301_train_risk_stability_routing"
MULTIROCKET_ROOT = DATA_ROOT / "exp249_250_official_mini_multirocket"
STRUCTURAL_ROOT = DATA_ROOT / "exp254_sktime_structural_features"
CHANGE_ROOT = DATA_ROOT / "exp255_change_point_features"
BASELINE_PATH = DATA_ROOT / "exp298_exp201_current_hard_rebase" / "exp298_frozen_predictions.jsonl"
BASELINE_SHA256 = "9de5eb0cd3226b0734e26179e246ad3fa59b3d95c472d78b424af5e80b65a727"
EXP300_PATH = DATA_ROOT / "exp300_independent_evidence_consensus" / "exp300_frozen_predictions.jsonl"
EXP300_SHA256 = "330e1397423967717f482b6660db5da4fee59cc9d76f052333d9175cf2b4a2d0"
EXPECTED_DATASETS = 251


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, separators=(",", ":")) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _load_cache(dataset: str) -> dict[str, np.ndarray]:
    multi_path, multi_meta_path = multirocket_cache_paths(MULTIROCKET_ROOT, 250, dataset)
    structure_path, structure_meta_path = structural_cache_paths(STRUCTURAL_ROOT, dataset)
    change_path, change_meta_path = change_point_cache_paths(CHANGE_ROOT, dataset)
    for archive, metadata_path in (
        (multi_path, multi_meta_path), (structure_path, structure_meta_path), (change_path, change_meta_path)
    ):
        if not archive.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing source cache for {dataset}: {archive}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("dataset")) != dataset:
            raise ValueError(f"cache dataset mismatch: {dataset}")
        if metadata.get("abstained"):
            raise ValueError(f"source cache abstained unexpectedly: {dataset}")
        if sha256_file(archive) != metadata.get("cache_sha256"):
            raise ValueError(f"source cache hash mismatch: {archive}")
    with np.load(multi_path, allow_pickle=False) as archive:
        multi_train = np.asarray(archive["kernels_1024__train_by_seed"], dtype=float)
        multi_test = np.asarray(archive["kernels_1024__test_by_seed"], dtype=float)
    with np.load(structure_path, allow_pickle=False) as archive:
        structural_train = np.asarray(archive["structural_combined_440__train_oof"], dtype=float)
        structural_test = np.asarray(archive["structural_combined_440__test_scores"], dtype=float)
    with np.load(change_path, allow_pickle=False) as archive:
        change_train = np.asarray(archive["change_point_combined_33__train_oof"], dtype=float)
        change_test = np.asarray(archive["change_point_combined_33__test_scores"], dtype=float)
    return {
        "multi_train": multi_train,
        "multi_test": multi_test,
        "structural_train": structural_train,
        "structural_test": structural_test,
        "change_train": change_train,
        "change_test": change_test,
    }


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(BASELINE_PATH) != BASELINE_SHA256:
        raise ValueError("Exp298 baseline hash mismatch")
    if sha256_file(EXP300_PATH) != EXP300_SHA256:
        raise ValueError("Exp300 prediction hash mismatch")
    baselines = _load_jsonl(BASELINE_PATH)
    exp300_rows = _load_jsonl(EXP300_PATH)
    if len(baselines) != EXPECTED_DATASETS or len({str(row["dataset"]) for row in baselines}) != EXPECTED_DATASETS:
        raise ValueError("Exp298 baseline coverage mismatch")
    exp300 = {(str(row["dataset"]), float(row["alpha"])): row for row in exp300_rows}
    if len(exp300) != EXPECTED_DATASETS * 4:
        raise ValueError("Exp300 source coverage mismatch")

    cache_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    risk_rows: list[dict[str, object]] = []
    total_train = 0
    total_false = 0
    for baseline in baselines:
        dataset = str(baseline["dataset"])
        cache = _load_cache(dataset)
        cache_by_dataset[dataset] = cache
        n_train = cache["multi_train"].shape[1]
        if n_train != int(baseline["n_train"]):
            raise ValueError(f"n_train mismatch: {dataset}")
        for alpha in (AUTOMATIC_ALPHA, PRIORITY_ALPHA):
            flags = consensus_train_flags(
                cache["multi_train"], cache["structural_train"], cache["change_train"], alpha=alpha
            )
            false_count = int(np.count_nonzero(flags["three_of_three"]))
            risk_rows.append({
                "dataset": dataset,
                "n_train": n_train,
                "alpha": alpha,
                "train_normal_three_of_three_count": false_count,
                "train_normal_three_of_three_rate": false_count / n_train,
                "train_normal_wilson_upper_95": wilson_upper(false_count, n_train),
                "uses_test_labels": False,
            })
            if alpha == AUTOMATIC_ALPHA:
                total_train += n_train
                total_false += false_count
    pooled_rate = total_false / total_train if total_train else 1.0
    pooled_upper = wilson_upper(total_false, total_train)
    risk_contract_pass = pooled_rate <= TRAIN_RISK_RATE_MAXIMUM and pooled_upper <= TRAIN_RISK_UPPER_MAXIMUM

    predictions: list[dict[str, object]] = []
    for baseline in baselines:
        dataset = str(baseline["dataset"])
        cache = cache_by_dataset[dataset]
        general = set(exp300[(dataset, GENERAL_ALPHA)]["residual_2of3_indices"])
        automatic = set(exp300[(dataset, AUTOMATIC_ALPHA)]["residual_3of3_indices"])
        priority = set(exp300[(dataset, PRIORITY_ALPHA)]["residual_3of3_indices"])
        automatic_stability = bootstrap_consensus_stability(
            dataset=dataset,
            multirocket_train_by_seed=cache["multi_train"], multirocket_test_by_seed=cache["multi_test"],
            structural_train=cache["structural_train"], structural_test=cache["structural_test"],
            change_point_train=cache["change_train"], change_point_test=cache["change_test"],
            candidate_indices=automatic, alpha=AUTOMATIC_ALPHA,
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
        )
        priority_stability = bootstrap_consensus_stability(
            dataset=dataset,
            multirocket_train_by_seed=cache["multi_train"], multirocket_test_by_seed=cache["multi_test"],
            structural_train=cache["structural_train"], structural_test=cache["structural_test"],
            change_point_train=cache["change_train"], change_point_test=cache["change_test"],
            candidate_indices=priority, alpha=PRIORITY_ALPHA,
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
        )
        row = build_routing_row(
            dataset=dataset, n_train=int(baseline["n_train"]), n_test=int(baseline["n_test"]),
            current_hard_indices=baseline["current_hard_indices"],
            general_candidates=general, automatic_candidates=automatic, priority_candidates=priority,
            automatic_stability=automatic_stability, priority_stability=priority_stability,
            automatic_risk_contract_pass=risk_contract_pass,
        )
        row.update({
            "pooled_train_normal_trials": total_train,
            "pooled_train_normal_false_count": total_false,
            "pooled_train_normal_false_rate": pooled_rate,
            "pooled_train_normal_wilson_upper_95": pooled_upper,
        })
        predictions.append(row)
    predictions.sort(key=lambda row: str(row["dataset"]))
    _write_csv(root / "exp301_train_risk_audit.csv", risk_rows)
    return freeze_prediction_rows(
        predictions, root / "exp301_frozen_predictions.jsonl", root / "exp301_prediction_manifest.json",
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "baseline_sha256": BASELINE_SHA256,
            "exp300_sha256": EXP300_SHA256,
            "datasets": EXPECTED_DATASETS,
            "rows": EXPECTED_DATASETS,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "automatic_risk_contract": {
                "rate_maximum": TRAIN_RISK_RATE_MAXIMUM,
                "wilson_upper_maximum": TRAIN_RISK_UPPER_MAXIMUM,
                "observed_rate": pooled_rate,
                "observed_wilson_upper": pooled_upper,
                "passed": risk_contract_pass,
            },
            "autonomous_policy_changed": False,
            "retrospective": True,
        },
    )


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp301_frozen_predictions.jsonl"
    predictions = verify_prediction_rows(prediction_path, root / "exp301_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp301 prediction coverage mismatch")
    labels = {str(row["dataset"]): load_test_labels_for_evaluation(str(row["dataset"])) for row in predictions}
    evaluated = evaluate_predictions(predictions, labels)
    summaries = summarize_evaluated(evaluated)
    _write_csv(root / "exp301_results.csv", evaluated)
    _write_csv(root / "exp301_summary.csv", summaries)
    summary = summaries[0]
    (root / "exp301_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "test_metrics_selected_policy": False,
        "autonomous_policy_changed": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Exp301 TRAIN Risk And Stability Routing",
        "",
        "Exp292 autonomous Hard alerts remain unchanged. All new lanes are retrospective shadow/review outputs.",
        "",
        "## TRAIN-normal automatic qualification contract",
        "",
        f"- Passed: {summary['automatic_risk_contract_pass']}",
        f"- Automatic qualification shadow: {summary['automatic_qualification_count']} (TP {summary['automatic_qualification_tp']}, FP {summary['automatic_qualification_fp']}, precision {float(summary['automatic_qualification_precision']):.3%})",
        f"- Priority review: {summary['priority_review_count']} (TP {summary['priority_review_tp']}, FP {summary['priority_review_fp']}, precision {float(summary['priority_review_precision']):.3%})",
        f"- General review: {summary['general_review_count']} (TP {summary['general_review_tp']}, FP {summary['general_review_fp']}, precision {float(summary['general_review_precision']):.3%})",
        "",
        "Alpha choices are retrospective research choices from Exp300. No candidate is promoted to autonomous Hard by this experiment.",
    ]
    (root / "RESULT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summaries


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None,
        errors=errors or [], timestamps=[], stdout_log=root / "logs" / f"{EXPERIMENT_ID}.log", started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp292 automatic Hard unchanged; alpha .01 qualification is shadow only",
        "review_metric_scope": "alpha .02 3-of-3 priority; alpha .005 2-of-3 general review",
        "retrospective": True,
        "prospective": False,
    })
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()
    _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"):
            run_prediction(args.output_root)
        if args.phase in ("evaluate", "all"):
            run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"])
        raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()
