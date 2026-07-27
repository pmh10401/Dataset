from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from experiments.exp393_396.artifact_io import load_test_labels
from experiments.exp397_399.artifact_io import (
    MANIFEST_NAME,
    PREDICTION_NAME,
    load_verified_ranking,
    sha256_file,
)
from experiments.exp397_399.contracts import WORKLOADS


def precision_at_k(
    ranked_rows: list[dict[str, Any]],
    workload_k: int,
    *,
    total_review_tp: int,
) -> dict[str, float | int]:
    selected = sorted(ranked_rows, key=lambda row: int(row["rank"]))[
        : int(workload_k)
    ]
    tp = sum(bool(row["is_true_anomaly"]) for row in selected)
    reviewed = len(selected)
    fp = reviewed - tp
    return {
        "workload_k": int(workload_k),
        "reviewed": reviewed,
        "tp_at_k": int(tp),
        "fp_at_k": int(fp),
        "precision_at_k": tp / reviewed if reviewed else 0.0,
        "review_tp_capture_rate": (
            tp / total_review_tp if total_review_tp else 0.0
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["variant"]), float(row["alpha"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def run_evaluation(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root)
    prediction_rows, prediction_manifest = load_verified_ranking(
        output_root / PREDICTION_NAME,
        output_root / MANIFEST_NAME,
    )
    dataset_names = {str(row["dataset"]) for row in prediction_rows}
    labels_map = load_test_labels(dataset_names)

    evaluated_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        dataset = str(row["dataset"])
        index = int(row["candidate_index"])
        labels = labels_map[dataset]
        if index < 0 or index >= len(labels):
            raise ValueError(f"candidate index out of range for {dataset}")
        evaluated_rows.append(
            {
                **row,
                "is_true_anomaly": str(labels[index]) not in {"0", "normal"},
            }
        )

    grouped = _group_rows(evaluated_rows)
    summary_rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    baseline_metrics: dict[tuple[float, str, int], float] = {}

    for (variant, alpha), rows in sorted(grouped.items()):
        total_review_tp = sum(bool(row["is_true_anomaly"]) for row in rows)
        total_review_fp = len(rows) - total_review_tp
        for workload in WORKLOADS:
            metrics = precision_at_k(
                rows,
                workload,
                total_review_tp=total_review_tp,
            )
            summary = {
                "variant": variant,
                "alpha": alpha,
                "workload_type": "global_k",
                **metrics,
                "total_standard_review_candidates": len(rows),
                "total_standard_review_tp": total_review_tp,
                "total_standard_review_fp": total_review_fp,
            }
            summary_rows.append(summary)
            if variant == "exp292_baseline":
                baseline_metrics[(alpha, "global_k", workload)] = float(
                    metrics["precision_at_k"]
                )

        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
        top_rows: list[dict[str, Any]] = []
        for dataset in sorted(labels_map):
            candidates = sorted(
                by_dataset.get(dataset, []),
                key=lambda row: int(row["rank"]),
            )
            selected = candidates[:1]
            top_rows.extend(selected)
            dataset_rows.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "dataset": dataset,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_index": (
                        int(selected[0]["candidate_index"]) if selected else ""
                    ),
                    "selected_rank": int(selected[0]["rank"]) if selected else "",
                    "selected_is_true_anomaly": (
                        bool(selected[0]["is_true_anomaly"]) if selected else ""
                    ),
                }
            )
        top_tp = sum(bool(row["is_true_anomaly"]) for row in top_rows)
        top_reviewed = len(top_rows)
        top_summary = {
            "variant": variant,
            "alpha": alpha,
            "workload_type": "per_dataset_top1",
            "workload_k": 1,
            "reviewed": top_reviewed,
            "tp_at_k": top_tp,
            "fp_at_k": top_reviewed - top_tp,
            "precision_at_k": top_tp / top_reviewed if top_reviewed else 0.0,
            "review_tp_capture_rate": (
                top_tp / total_review_tp if total_review_tp else 0.0
            ),
            "total_standard_review_candidates": len(rows),
            "total_standard_review_tp": total_review_tp,
            "total_standard_review_fp": total_review_fp,
        }
        summary_rows.append(top_summary)
        if variant == "exp292_baseline":
            baseline_metrics[(alpha, "per_dataset_top1", 1)] = float(
                top_summary["precision_at_k"]
            )

    for row in summary_rows:
        baseline = baseline_metrics[
            (
                float(row["alpha"]),
                str(row["workload_type"]),
                int(row["workload_k"]),
            )
        ]
        row["baseline_precision_at_k"] = baseline
        row["precision_lift_vs_exp292"] = float(row["precision_at_k"]) - baseline
        row["retrospective_counterfactual"] = True
        row["eligible_for_deployment"] = False
        row["test_metrics_select_policy"] = False

    candidate_path = output_root / "exp398_candidate_evaluation.csv"
    summary_path = output_root / "exp398_precision_at_k.csv"
    dataset_path = output_root / "exp398_dataset_diagnostics.csv"
    _write_csv(candidate_path, evaluated_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(dataset_path, dataset_rows)
    manifest = {
        "experiment_id": "experiment_398_locked_review_ranking_evaluation",
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "datasets": len(dataset_names),
        "candidate_rows": len(evaluated_rows),
        "summary_rows": len(summary_rows),
        "labels_loaded_after_prediction_verification": True,
        "candidate_evaluation_sha256": sha256_file(candidate_path),
        "precision_at_k_sha256": sha256_file(summary_path),
        "dataset_diagnostics_sha256": sha256_file(dataset_path),
        "test_metrics_select_policy": False,
        "eligible_for_deployment": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    (output_root / "exp398_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

