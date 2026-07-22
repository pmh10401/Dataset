from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp226_235_hard_alert_quality.preprocessing import preprocessing_contract
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows

from .config import EXPECTED_DATASETS, EXPERIMENT_ID, INPUTS, OUTPUT_ROOT
from .routing import build_componentized_row


def _load_inputs() -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, str]]:
    loaded: dict[str, dict[str, dict[str, object]]] = {}
    hashes: dict[str, str] = {}
    for name, (path, expected_hash) in INPUTS.items():
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ValueError(f"{name} prediction SHA256 mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        by_dataset = {str(row["dataset"]): row for row in rows}
        if len(rows) != EXPECTED_DATASETS or len(by_dataset) != EXPECTED_DATASETS:
            raise ValueError(f"{name} coverage mismatch")
        loaded[name] = by_dataset
        hashes[name] = observed
    populations = [set(rows) for rows in loaded.values()]
    if any(values != populations[0] for values in populations[1:]):
        raise ValueError("input population mismatch")
    return loaded, hashes


def run_prediction(root: Path) -> dict[str, object]:
    inputs, hashes = _load_inputs()
    predictions = [
        build_componentized_row(
            dataset=dataset,
            exp236=inputs["exp236"][dataset],
            exp253=inputs["exp253"][dataset],
            exp290=inputs["exp290"][dataset],
            exp291=inputs["exp291"][dataset],
        )
        for dataset in sorted(inputs["exp291"])
    ]
    return freeze_rows(
        predictions,
        root / "exp306_frozen_predictions.jsonl",
        root / "exp306_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "datasets": EXPECTED_DATASETS,
            "input_sha256": hashes,
            "preprocessing_contract": preprocessing_contract(),
            "candidate_operational_policy_only": True,
            "operational_policy_updated": False,
            "research_extensions_autonomous": False,
            "retrospective": True,
            "prospective_validated": False,
        },
    )


def _metrics(indices: list[int], labels: np.ndarray) -> tuple[int, int]:
    tp = sum(int(labels[int(index)] == 1) for index in indices)
    return tp, len(indices) - tp


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    path = root / "exp306_frozen_predictions.jsonl"
    predictions = verify_rows(path, root / "exp306_prediction_manifest.json")
    lane_keys = (
        "core_hard_indices", "exp236_research_shadow_indices",
        "exp253_research_shadow_indices", "benchmark_union_hard_indices",
    )
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in predictions
    }
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    truth_total = sum(int(np.count_nonzero(values == 1)) for values in labels.values())
    for key in lane_keys:
        total_tp = total_fp = 0
        dataset_f1: list[float] = []
        for row in predictions:
            dataset = str(row["dataset"]); y = labels[dataset]
            tp, fp = _metrics(row[key], y)
            truth = int(np.count_nonzero(y == 1)); fn = truth - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / truth if truth else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            total_tp += tp; total_fp += fp; dataset_f1.append(f1)
            detail.append({
                "experiment_id": EXPERIMENT_ID, "dataset": dataset, "lane": key,
                "count": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
            })
        count = total_tp + total_fp
        precision = total_tp / count if count else 0.0
        recall = total_tp / truth_total if truth_total else 0.0
        summary.append({
            "experiment_id": EXPERIMENT_ID, "lane": key, "datasets": len(predictions),
            "count": count, "tp": total_tp, "fp": total_fp,
            "precision": precision, "recall": recall,
            "mean_dataset_f1": float(np.mean(dataset_f1)),
            "autonomous": key == "core_hard_indices",
            "research_only": key in {"exp236_research_shadow_indices", "exp253_research_shadow_indices"},
            "operational_policy_updated": False,
            "retrospective": True,
        })
    _write_csv(root / "exp306_results.csv", detail)
    _write_csv(root / "exp306_summary.csv", summary)
    report = [
        "# Exp306 Componentized Operational Routing", "",
        "Exp291 is decomposed without changing its frozen benchmark union.", "",
        "| Lane | Count | TP | FP | Precision | Recall | Mean F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['lane']} | {row['count']} | {row['tp']} | {row['fp']} | "
            f"{100*float(row['precision']):.3f}% | {100*float(row['recall']):.3f}% | {float(row['mean_dataset_f1']):.6f} |"
        )
    report.extend(["", "- Exp236/253 extensions are retrospective research Shadow, not autonomous alerts.", "- The current Exp291 benchmark artifact is unchanged.", ""])
    (root / "RESULT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (root / "exp306_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID, "prediction_sha256": sha256_file(path),
        "labels_loaded_after_prediction_freeze": True,
        "operational_policy_updated": False, "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors=None) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"; log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None, errors=errors or [], timestamps=[], stdout_log=log, started_unix=started)
    payload.update({"autonomous_metric_scope": "Guarded Exp137 core candidate; policy not deployed", "review_metric_scope": "Exp236/253 retrospective research Shadow", "retrospective": True, "prospective": False})
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all"); parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT); args = parser.parse_args()
    started = time.time(); _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"): run_prediction(args.output_root)
        if args.phase in ("evaluate", "all"): run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"]); raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__": main()
