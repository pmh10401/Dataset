#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiments.exp171_172.run_experiment_171_172_train_risk_policy import (
    read_csv,
    read_jsonl,
    sha256_file,
    verify_prediction_manifest,
    write_json,
)


def _close(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def reaggregate(output_dir: Path, *, write_result: bool = True) -> dict:
    output_dir = Path(output_dir)
    manifest, prediction_path, policy = verify_prediction_manifest(output_dir)
    predictions = read_jsonl(prediction_path)
    metrics = read_csv(output_dir / "06_exp172_dataset_metrics.csv")
    summary_rows = read_csv(output_dir / "07_exp172_global_summary.csv")
    if len(summary_rows) != 1:
        raise ValueError("Exp172 global summary must contain exactly one row")
    summary = summary_rows[0]
    mismatches = []

    prediction_by_dataset = {row["dataset"]: row for row in predictions}
    metric_by_dataset = {row["dataset"]: row for row in metrics}
    if len(prediction_by_dataset) != len(predictions):
        mismatches.append("duplicate prediction dataset")
    if len(metric_by_dataset) != len(metrics):
        mismatches.append("duplicate metric dataset")
    if set(prediction_by_dataset) != set(metric_by_dataset):
        mismatches.append("prediction/metric dataset coverage")
    if len(predictions) != int(manifest["dataset_count"]):
        mismatches.append("manifest dataset count")

    selected_key = tuple(policy["selected"]["policy_key"])
    for dataset, prediction in prediction_by_dataset.items():
        key = (
            prediction["pool_policy"],
            float(prediction["alpha"]),
            prediction["source_config"],
        )
        if key != (selected_key[0], float(selected_key[1]), selected_key[2]):
            mismatches.append(f"selected policy key: {dataset}")
        indices = [int(index) for index in prediction["candidate_indices"]]
        if len(indices) != int(prediction["candidate_count"]):
            mismatches.append(f"candidate count: {dataset}")
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= int(prediction["n_test"]) for index in indices
        ):
            mismatches.append(f"candidate index validity: {dataset}")
        metric = metric_by_dataset.get(dataset)
        if metric and int(metric["review_candidates"]) != len(indices):
            mismatches.append(f"prediction/metric candidate count: {dataset}")
        if metric and int(metric["tp"]) + int(metric["fp"]) != len(indices):
            mismatches.append(f"TP+FP candidate count: {dataset}")

    tp = sum(int(row["tp"]) for row in metrics)
    fp = sum(int(row["fp"]) for row in metrics)
    review_candidates = sum(int(row["review_candidates"]) for row in metrics)
    mean_recall = float(np.mean([float(row["recall"]) for row in metrics])) if metrics else 0.0
    mean_f1 = float(np.mean([float(row["f1"]) for row in metrics])) if metrics else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    expected_values = {
        "datasets": len(metrics),
        "review_candidates": review_candidates,
        "tp": tp,
        "fp": fp,
        "micro_precision": precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "zero_f1_datasets": sum(float(row["f1"]) == 0.0 for row in metrics),
        "candidate_zero_datasets": sum(int(row["review_candidates"]) == 0 for row in metrics),
    }
    for field, expected in expected_values.items():
        actual = summary[field]
        if isinstance(expected, float):
            if not _close(actual, expected):
                mismatches.append(f"summary {field}")
        elif int(actual) != expected:
            mismatches.append(f"summary {field}")

    result = {
        **expected_values,
        "mismatch_count": len(mismatches),
        "mismatches": sorted(mismatches),
        "policy_sha256": manifest["policy_sha256"],
        "prediction_sha256": manifest["prediction_sha256"],
        "dataset_metrics_sha256": sha256_file(output_dir / "06_exp172_dataset_metrics.csv"),
        "global_summary_sha256": sha256_file(output_dir / "07_exp172_global_summary.csv"),
        "retrospective": True,
        "autonomous_metric_scope": "not_evaluated",
    }
    if mismatches:
        raise ValueError(f"independent reaggregation mismatch: {sorted(mismatches)}")
    if write_result:
        write_json(output_dir / "11_independent_reaggregation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/exp171_172_train_risk_policy"),
    )
    args = parser.parse_args()
    result = reaggregate(args.output_dir)
    print(f"Exp171/172 independent verification passed: {result}")


if __name__ == "__main__":
    main()
