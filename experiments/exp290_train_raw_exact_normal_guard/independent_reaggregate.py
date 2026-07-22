from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .contracts import sha256_file


VIEWS = ("baseline_hard", "guarded_hard", "train_raw_exact_normal")


def reaggregate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    result: dict[str, object] = {"datasets": len(rows)}
    for view in VIEWS:
        for metric in ("count", "tp", "fp"):
            result[f"{view}_{metric}"] = sum(int(row[f"{view}_{metric}"]) for row in rows)
        tp = int(result[f"{view}_tp"])
        fp = int(result[f"{view}_fp"])
        result[f"{view}_precision"] = tp / (tp + fp) if tp + fp else 0.0
        if rows and f"{view}_f1" in rows[0]:
            result[f"{view}_mean_f1"] = float(np.mean([float(row[f"{view}_f1"]) for row in rows]))
    for row in rows:
        if int(row["baseline_hard_count"]) != int(row["guarded_hard_count"]) + int(row["train_raw_exact_normal_count"]):
            raise ValueError("dataset Hard partition count mismatch")
        if int(row["baseline_hard_tp"]) != int(row["guarded_hard_tp"]) + int(row["train_raw_exact_normal_tp"]):
            raise ValueError("dataset Hard partition TP mismatch")
        if int(row["baseline_hard_fp"]) != int(row["guarded_hard_fp"]) + int(row["train_raw_exact_normal_fp"]):
            raise ValueError("dataset Hard partition FP mismatch")
    result["alert_reduction"] = int(result["baseline_hard_count"]) - int(result["guarded_hard_count"])
    result["tp_reduction"] = int(result["baseline_hard_tp"]) - int(result["guarded_hard_tp"])
    result["fp_reduction"] = int(result["baseline_hard_fp"]) - int(result["guarded_hard_fp"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    results_path = root / "exp290_results.csv"
    rows = list(csv.DictReader(results_path.open(encoding="utf-8")))
    aggregate = reaggregate_rows(rows)
    summary = next(csv.DictReader((root / "exp290_summary.csv").open(encoding="utf-8")))
    for key in (
        "datasets", "baseline_hard_count", "baseline_hard_tp", "baseline_hard_fp",
        "guarded_hard_count", "guarded_hard_tp", "guarded_hard_fp",
        "train_raw_exact_normal_count", "train_raw_exact_normal_tp", "train_raw_exact_normal_fp",
    ):
        if int(aggregate[key]) != int(summary[key]):
            raise ValueError(f"independent summary mismatch: {key}")
    aggregate.update(
        {
            "results_path": str(results_path),
            "results_sha256": sha256_file(results_path),
            "summary_sha256": sha256_file(root / "exp290_summary.csv"),
            "prediction_sha256": sha256_file(root / "exp290_frozen_predictions.jsonl"),
            "independently_reaggregated": True,
            "retrospective_counterfactual": True,
        }
    )
    (root / "exp290_independent_reaggregation.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
