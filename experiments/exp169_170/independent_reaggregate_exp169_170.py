#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments.exp169_170.run_experiment_169_170_oob_pooled_calibration import (
    read_jsonl,
    sha256_file,
    validate_prediction_grid,
    verify_prediction_manifest,
)

POLICIES = ("similarity_pool", "coarse_strata_pool", "global_balanced_pool")
ALPHAS = (0.005, 0.01, 0.02, 0.05)
SOURCES = ("spectral", "shapelet", "interval")
SOURCE_PAIRS = {
    "spectral+shapelet": ("spectral", "shapelet"),
    "spectral+interval": ("spectral", "interval"),
    "shapelet+interval": ("shapelet", "interval"),
}
SOURCE_CONFIGS = (*SOURCE_PAIRS, "2of3", "3of3")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_key(row: dict) -> tuple[str, str, float, str]:
    return (
        str(row["dataset"]),
        str(row["pool_policy"]),
        float(row["alpha"]),
        str(row["source_config"]),
    )


def validate_metric_prediction_alignment(
    predictions: list[dict],
    metrics: list[dict],
) -> None:
    prediction_map = {_row_key(row): row for row in predictions}
    metric_map = {_row_key(row): row for row in metrics}
    if len(prediction_map) != len(predictions):
        raise ValueError("duplicate prediction key")
    if len(metric_map) != len(metrics):
        raise ValueError("duplicate metric key")
    if set(prediction_map) != set(metric_map):
        raise ValueError("prediction and metric key coverage mismatch")

    for key, prediction in prediction_map.items():
        metric = metric_map[key]
        candidate_count = int(prediction["candidate_count"])
        indices = [int(value) for value in prediction["candidate_indices"]]
        n_test = int(prediction["n_test"])
        if candidate_count != len(indices) or len(indices) != len(set(indices)):
            raise ValueError(f"candidate count or duplicate mismatch for {key}")
        if any(index < 0 or index >= n_test for index in indices):
            raise ValueError(f"candidate index out of range for {key}")
        if int(metric["n_test"]) != n_test:
            raise ValueError(f"metric TEST length mismatch for {key}")
        if int(metric["predicted_count"]) != candidate_count:
            raise ValueError(f"metric prediction count mismatch for {key}")
        if int(metric["tp"]) + int(metric["fp"]) != candidate_count:
            raise ValueError(f"TP plus FP does not equal candidate count for {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs/exp169_170_oob_pooled_calibration")
    args = parser.parse_args()

    metrics_path = args.output_dir / "08_exp170_dataset_metrics.csv"
    summary_path = args.output_dir / "09_exp170_global_summary.csv"

    if not metrics_path.exists():
        print(f"Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)
    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    metrics = read_csv(metrics_path)
    summary = read_csv(summary_path)
    manifest, prediction_path = verify_prediction_manifest(args.output_dir)
    predictions = read_jsonl(prediction_path)
    validate_prediction_grid(predictions, expected_datasets=int(manifest["dataset_count"]))
    validate_metric_prediction_alignment(predictions, metrics)

    # 1. Verify row counts and structures
    # Grid of configurations: 3 policies * 4 alphas * 5 configs = 60 global rows
    if len(summary) != 60:
        print(f"Expected 60 summary rows, got {len(summary)}", file=sys.stderr)
        sys.exit(1)

    # Group dataset metrics by (policy, alpha, source_config)
    grouped = {}
    for r in metrics:
        key = (r["pool_policy"], float(r["alpha"]), r["source_config"])
        grouped.setdefault(key, []).append(r)

    mismatch_count = 0
    verification_results = []

    for row in summary:
        policy = row["pool_policy"]
        alpha = float(row["alpha"])
        source_config = row["source_config"]
        key = (policy, alpha, source_config)

        if key not in grouped:
            print(f"Group not found in dataset metrics for summary row: {key}", file=sys.stderr)
            mismatch_count += 1
            continue

        group = grouped[key]

        # Recompute values
        datasets_count = len(group)
        review_candidates = sum(int(r["predicted_count"]) for r in group)
        tp = sum(int(r["tp"]) for r in group)
        fp = sum(int(r["fp"]) for r in group)
        micro_precision = tp / (tp + fp) if tp + fp else 0.0
        mean_recall = float(np.mean([float(r["recall"]) for r in group]))
        mean_f1 = float(np.mean([float(r["f1"]) for r in group]))
        zero_f1_datasets = sum(float(r["f1"]) == 0.0 for r in group)
        candidate_zero_datasets = sum(int(r["predicted_count"]) == 0 for r in group)
        mean_candidates_per_dataset = float(np.mean([int(r["predicted_count"]) for r in group]))
        resolution_blocked_datasets = sum(int(r["resolution_blocked"]) for r in group)

        # Compare with summary row
        # Help function to assert close floats or equal ints
        def verify_field(field_name: str, computed: float | int, reported_str: str, is_float: bool = False) -> bool:
            reported = float(reported_str) if is_float else int(reported_str)
            if is_float:
                close = math.isclose(computed, reported, abs_tol=1e-6)
                if not close:
                    print(f"Mismatch in {key} for {field_name}: computed={computed}, reported={reported}", file=sys.stderr)
                    return False
            else:
                if computed != reported:
                    print(f"Mismatch in {key} for {field_name}: computed={computed}, reported={reported}", file=sys.stderr)
                    return False
            return True

        checks = [
            verify_field("datasets", datasets_count, row["datasets"]),
            verify_field("review_candidates", review_candidates, row["review_candidates"]),
            verify_field("tp", tp, row["tp"]),
            verify_field("fp", fp, row["fp"]),
            verify_field("micro_precision", micro_precision, row["micro_precision"], is_float=True),
            verify_field("mean_recall", mean_recall, row["mean_recall"], is_float=True),
            verify_field("mean_f1", mean_f1, row["mean_f1"], is_float=True),
            verify_field("zero_f1_datasets", zero_f1_datasets, row["zero_f1_datasets"]),
            verify_field("candidate_zero_datasets", candidate_zero_datasets, row["candidate_zero_datasets"]),
            verify_field("mean_candidates_per_dataset", mean_candidates_per_dataset, row["mean_candidates_per_dataset"], is_float=True),
            verify_field("resolution_blocked_datasets", resolution_blocked_datasets, row["resolution_blocked_datasets"]),
        ]

        if not all(checks):
            mismatch_count += 1
            verification_results.append({
                "key": list(key),
                "status": "mismatch",
            })
        else:
            verification_results.append({
                "key": list(key),
                "status": "verified",
            })

    reaggregation_json = {
        "status": "success" if mismatch_count == 0 else "failure",
        "mismatch_count": mismatch_count,
        "dataset_count": int(manifest["dataset_count"]),
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "prediction_sha256": manifest["prediction_sha256"],
        "metrics_sha256": sha256_file(metrics_path),
        "summary_sha256": sha256_file(summary_path),
        "results": verification_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_path = args.output_dir / "12_independent_reaggregation.json"
    write_path.write_text(json.dumps(reaggregation_json, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if mismatch_count > 0:
        print(f"Independent reaggregation failed with {mismatch_count} mismatches.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Independent reaggregation verified successfully.", flush=True)


if __name__ == "__main__":
    main()
