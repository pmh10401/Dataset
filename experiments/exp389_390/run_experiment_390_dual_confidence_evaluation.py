from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp389_390.dual_confidence_core import (
    EXPECTED_DATASETS,
    evaluate_indices,
    load_frozen_predictions,
    sha256_file,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_ROOT / "univariate_ts.db"
SOURCE_ROOT = DATA_ROOT / "exp386_388_existing_evidence_alpha_sweep"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "exp389_390_dual_confidence_tiers"
TIER_KEYS = {
    "hard": "automatic_hard_indices",
    "highest": "highest_confidence_indices",
    "extended": "extended_confidence_indices",
    "general": "general_confidence_indices",
}


def summarize_detail_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty evaluation rows")
    summary: dict[str, int | float | bool] = {
        "datasets": len(rows),
        "retrospective_counterfactual": True,
        "prospective_validated": False,
        "test_metrics_select_policy": False,
    }
    for tier in TIER_KEYS:
        count = sum(int(row[f"{tier}_count"]) for row in rows)
        tp = sum(int(row[f"{tier}_tp"]) for row in rows)
        fp = sum(int(row[f"{tier}_fp"]) for row in rows)
        summary[f"{tier}_count"] = count
        summary[f"{tier}_tp"] = tp
        summary[f"{tier}_fp"] = fp
        summary[f"{tier}_precision"] = tp / (tp + fp) if tp + fp else 0.0
    summary["mean_hard_f1"] = sum(float(row["hard_f1"]) for row in rows) / len(rows)
    summary["alpha010_strong_count"] = int(summary["highest_count"]) + int(
        summary["extended_count"]
    )
    summary["alpha010_strong_tp"] = int(summary["highest_tp"]) + int(
        summary["extended_tp"]
    )
    summary["alpha010_strong_fp"] = int(summary["highest_fp"]) + int(
        summary["extended_fp"]
    )
    strong_count = int(summary["alpha010_strong_count"])
    summary["alpha010_strong_precision"] = (
        int(summary["alpha010_strong_tp"]) / strong_count if strong_count else 0.0
    )
    return summary


def load_test_labels(dataset_names: set[str]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = defaultdict(list)
    uri = f"file:{DB_PATH}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        for dataset, label in connection.execute(
            """
            SELECT d.name, i.label
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id
            WHERE i.split = 'TEST'
            ORDER BY d.name, i.instance_index
            """
        ):
            if dataset in dataset_names:
                labels[str(dataset)].append(str(label))
    missing = dataset_names - set(labels)
    if missing:
        raise ValueError(f"missing TEST labels for {len(missing)} datasets")
    return dict(labels)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_full_baseline(summary: dict) -> None:
    baseline_path = SOURCE_ROOT / "alpha_0p005" / "exp358_summary.csv"
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline_rows = list(csv.DictReader(handle))
    if len(baseline_rows) != 1:
        raise ValueError("invalid frozen Exp358 baseline summary")
    baseline = baseline_rows[0]
    exact = {
        "hard_count": "hard_count",
        "hard_tp": "hard_tp",
        "hard_fp": "hard_fp",
    }
    for current_key, baseline_key in exact.items():
        if int(summary[current_key]) != int(baseline[baseline_key]):
            raise ValueError(f"Hard baseline mismatch: {current_key}")
    floats = {
        "hard_precision": "hard_precision",
        "mean_hard_f1": "hard_mean_f1",
    }
    for current_key, baseline_key in floats.items():
        if abs(float(summary[current_key]) - float(baseline[baseline_key])) > 1e-12:
            raise ValueError(f"Hard baseline mismatch: {current_key}")


def write_status(root: Path, status: str, done: int, total: int, errors: int = 0) -> None:
    (root / "exp390_status.json").write_text(
        json.dumps(
            {
                "id": "experiment_390_dual_confidence_evaluation",
                "status": status,
                "phase": status,
                "datasets_done": done,
                "datasets_total": total,
                "progress_percent": 100.0 * done / total if total else 0.0,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run(output_root: Path) -> dict:
    prediction_path = output_root / "exp389_frozen_predictions.jsonl"
    manifest_path = output_root / "exp389_prediction_manifest.json"
    rows = load_frozen_predictions(prediction_path, manifest_path)
    write_status(output_root, "evaluating", 0, len(rows))

    labels_by_dataset = load_test_labels({str(row["dataset"]) for row in rows})
    details = []
    for number, row in enumerate(rows, start=1):
        labels = labels_by_dataset[str(row["dataset"])]
        if len(labels) != int(row["n_test"]):
            raise ValueError(f"TEST label length mismatch: {row['dataset']}")
        detail: dict[str, object] = {
            "dataset": row["dataset"],
            "n_test": row["n_test"],
            "true_anomaly_count": sum(label == "1" for label in labels),
        }
        for tier, key in TIER_KEYS.items():
            metrics = evaluate_indices(row[key], labels)
            detail.update({f"{tier}_{name}": value for name, value in metrics.items()})
        details.append(detail)
        if number == len(rows) or number % 100 == 0:
            write_status(output_root, "evaluating", number, len(rows))
            print(
                f"Progress: [{number}/{len(rows)}] rows={number} "
                f"dataset={row['dataset']} errors=0",
                flush=True,
            )

    summary = summarize_detail_rows(details)
    if len(rows) == EXPECTED_DATASETS:
        verify_full_baseline(summary)
    write_csv(output_root / "exp390_detail.csv", details)
    write_csv(output_root / "exp390_summary.csv", [summary])
    evaluation_manifest = {
        "experiment_id": "experiment_390_dual_confidence_evaluation",
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_manifest_sha256": sha256_file(manifest_path),
        "detail_sha256": sha256_file(output_root / "exp390_detail.csv"),
        "summary_sha256": sha256_file(output_root / "exp390_summary.csv"),
        "datasets": len(rows),
        "test_labels_loaded_after_prediction_hash_verification": True,
        "hard_lane_preserved": True,
        "test_metrics_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp390_evaluation_manifest.json").write_text(
        json.dumps(evaluation_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_status(output_root, "completed", len(rows), len(rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
