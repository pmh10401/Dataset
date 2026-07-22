from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation

VIEWS = ("baseline_hard", "final_hard", "promoted", "final_standard", "final_priority", "grade_s", "grade_a", "grade_b")


def verify_prediction_manifest(path: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["prediction_sha256"]:
        raise ValueError("prediction hash mismatch")
    return manifest


def metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    tp, fp, fn = len(selected & truth), len(selected - truth), len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def _paired_bootstrap_ci(values: list[float], *, seed: int = 20260720,
                         samples: int = 2000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(np.mean(rng.choice(array, size=len(array), replace=True)))
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def evaluate(output_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    path = output_root / "frozen_predictions.jsonl"
    manifest = verify_prediction_manifest(path, output_root / "prediction_manifest.json")
    evaluated = []
    field_map = {view: f"{view}_indices" for view in VIEWS}
    for row in map(json.loads, path.read_text().splitlines()):
        truth = set(np.flatnonzero(load_test_labels_for_evaluation(str(row["dataset"])) == 1).astype(int))
        output = dict(row); output["true_anomaly_count"] = len(truth)
        for view, field in field_map.items():
            output.update({f"{view}_{key}": value for key, value in metrics(set(map(int, row[field])), truth).items()})
        evaluated.append(output)
    summary: dict[str, object] = {
        "experiment_id": manifest["experiment_id"], "datasets": len(evaluated),
        "selected_from_prior_test_review": 1, "retrospective": 1, "deployment_eligible": 0,
    }
    for view in VIEWS:
        tp = sum(int(row[f"{view}_tp"]) for row in evaluated)
        fp = sum(int(row[f"{view}_fp"]) for row in evaluated)
        fn = sum(int(row[f"{view}_fn"]) for row in evaluated)
        summary.update({
            f"{view}_count": tp + fp, f"{view}_tp": tp, f"{view}_fp": fp, f"{view}_fn": fn,
            f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
            f"{view}_recall": tp / (tp + fn) if tp + fn else 0.0,
            f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in evaluated])),
        })
    summary["baseline_hard_zero_alert_datasets"] = sum(
        int(row["baseline_hard_count"] == 0) for row in evaluated
    )
    summary["final_hard_zero_alert_datasets"] = sum(
        int(row["final_hard_count"] == 0) for row in evaluated
    )
    summary["baseline_hard_zero_f1_datasets"] = sum(
        int(float(row["baseline_hard_f1"]) == 0.0) for row in evaluated
    )
    summary["final_hard_zero_f1_datasets"] = sum(
        int(float(row["final_hard_f1"]) == 0.0) for row in evaluated
    )
    summary["promoted_dataset_count"] = sum(int(row["promoted_count"] > 0) for row in evaluated)
    for metric in ("fp", "recall", "f1"):
        changes = [
            float(row[f"final_hard_{metric}"]) - float(row[f"baseline_hard_{metric}"])
            for row in evaluated
        ]
        low, high = _paired_bootstrap_ci(changes)
        summary[f"hard_{metric}_mean_change"] = float(np.mean(changes))
        summary[f"hard_{metric}_change_ci95_low"] = low
        summary[f"hard_{metric}_change_ci95_high"] = high
    summary["incremental_precision_gate_80pct"] = int(float(summary["promoted_precision"]) >= 0.80)
    _write_csv(output_root / "results.csv", evaluated)
    _write_csv(output_root / "summary.csv", [summary])
    transitions = [{"dataset": row["dataset"], "promoted_count": row["promoted_count"], "promoted_tp": row["promoted_tp"], "promoted_fp": row["promoted_fp"]} for row in evaluated]
    grades = [{"dataset": row["dataset"], **{f"{g}_{k}": row[f"{g}_{k}"] for g in ("grade_s", "grade_a", "grade_b") for k in ("count", "tp", "fp", "precision")}} for row in evaluated]
    _write_csv(output_root / "lane_transitions.csv", transitions)
    _write_csv(output_root / "confidence_grades.csv", grades)
    (output_root / "evaluation_manifest.json").write_text(json.dumps({"prediction_sha256": manifest["prediction_sha256"], "labels_used_after_prediction_freeze": True, "rows": len(evaluated)}, sort_keys=True, indent=2) + "\n")
    return evaluated, summary
