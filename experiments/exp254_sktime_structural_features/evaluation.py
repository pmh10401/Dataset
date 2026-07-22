from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from .config import EXPERIMENT


VIEWS = (
    "candidate", "hard_confirmation", "standard_priority", "priority_confirmation",
    "outside_lane", "exp253_hard_confirmation", "exp253_remaining_standard_priority",
)


def verify_prediction_manifest(path: Path, manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["prediction_sha256"]:
        raise ValueError("prediction hash mismatch")
    return manifest


def metrics(indices: list[int], truth: set[int]) -> dict[str, float | int]:
    selected = set(map(int, indices))
    tp, fp, fn = len(selected & truth), len(selected - truth), len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def evaluate_experiment(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    path = root / "frozen_predictions.jsonl"
    manifest = verify_prediction_manifest(path, root / "prediction_manifest.json")
    evaluated = []
    for row in map(json.loads, path.read_text().splitlines()):
        truth = set(np.flatnonzero(load_test_labels_for_evaluation(str(row["dataset"])) == 1).astype(int))
        output = dict(row); output["true_anomaly_count"] = len(truth)
        for view in VIEWS:
            output.update({f"{view}_{key}": value for key, value in metrics(row[f"{view}_indices"], truth).items()})
        evaluated.append(output)
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        groups[(str(row["variant"]), float(row["alpha"]))].append(row)
    summaries = []
    for (variant, alpha), group in sorted(groups.items()):
        summary: dict[str, object] = {
            "experiment_id": group[0]["experiment_id"], "experiment": EXPERIMENT,
            "source": group[0]["source"], "variant": variant, "alpha": alpha,
            "datasets": len(group),
            "abstained_datasets": sum(bool(row["calibration_abstained"]) for row in group),
            "resolution_blocked_datasets": sum(bool(row["resolution_blocked"]) for row in group),
            "hard_policy_changed": 0, "comparison_eligible": 0, "retrospective": 1,
        }
        for view in VIEWS:
            tp = sum(int(row[f"{view}_tp"]) for row in group)
            fp = sum(int(row[f"{view}_fp"]) for row in group)
            summary.update({
                f"{view}_count": tp + fp, f"{view}_tp": tp, f"{view}_fp": fp,
                f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
                f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in group])),
            })
        summary["confirmation_gate_passed"] = int(summary["hard_confirmation_count"] >= 100 and summary["hard_confirmation_precision"] >= 0.89546)
        summary["review_gate_passed"] = int(summary["standard_priority_count"] >= 50 and summary["standard_priority_precision"] >= 0.70)
        summary["exp253_incremental_80pct_gate"] = int(summary["exp253_remaining_standard_priority_count"] >= 1 and summary["exp253_remaining_standard_priority_precision"] >= 0.80)
        summary["gate_scope"] = "retrospective_research_gate_only"
        summaries.append(summary)
    _write_csv(root / "results.csv", evaluated); _write_csv(root / "summary.csv", summaries)
    (root / "evaluation_manifest.json").write_text(json.dumps({
        "prediction_sha256": manifest["prediction_sha256"], "rows": len(evaluated),
        "summary_rows": len(summaries), "contains_test_labels": True,
        "labels_used_after_prediction_freeze": True,
    }, sort_keys=True, indent=2) + "\n")
    return evaluated, summaries
