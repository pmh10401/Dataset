from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.exp393_396.artifact_io import load_test_labels

from experiments.exp400_402 import EXP401_ID
from experiments.exp400_402.artifact_io import (
    MANIFEST_NAME,
    PREDICTION_NAME,
    load_verified_prediction,
    sha256_file,
)
from experiments.exp400_402.contracts import (
    ALPHAS,
    CONTROL_VARIANTS,
    VISIBLE_VARIANTS,
)


WORKLOADS = (25, 50, 100, 200)
VARIANTS = VISIBLE_VARIANTS + CONTROL_VARIANTS


def _is_anomaly(value: object) -> bool:
    return str(value).strip().lower() not in {"0", "0.0", "normal", "false"}


def _rank_for(
    candidate: Mapping[str, object],
    variant: str,
    alpha_key: str,
) -> int:
    if variant == "exp292_baseline":
        return int(candidate["exp292_rank"])
    if variant == "tier_preserving_grammar":
        return int(candidate["tier_preserving_rank"][alpha_key])
    return int(candidate["control_rank"][variant][alpha_key])


def _spearman(
    baseline_by_key: Mapping[tuple[str, int], int],
    comparison_by_key: Mapping[tuple[str, int], int],
) -> float:
    keys = sorted(baseline_by_key)
    if keys != sorted(comparison_by_key):
        raise ValueError("rank comparison candidate universe mismatch")
    if len(keys) < 2:
        return 1.0
    left = [float(baseline_by_key[key]) for key in keys]
    right = [float(comparison_by_key[key]) for key in keys]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if not left_scale or not right_scale:
        return 1.0 if left == right else 0.0
    return numerator / (left_scale * right_scale)


def evaluate_rows(
    prediction_rows: Sequence[Mapping[str, object]],
    labels_by_dataset: Mapping[str, Sequence[object]],
    *,
    workloads: Sequence[int] = WORKLOADS,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    datasets = {str(row["dataset"]) for row in prediction_rows}
    if datasets != set(labels_by_dataset):
        raise ValueError("prediction and label dataset coverage differs")

    evaluated: list[dict[str, object]] = []
    for row in prediction_rows:
        dataset = str(row["dataset"])
        n_test = int(row["n_test"])
        labels = labels_by_dataset[dataset]
        if len(labels) != n_test:
            raise ValueError(f"TEST label length mismatch: {dataset}")
        for candidate in row["candidates"]:
            index = int(candidate["candidate_index"])
            for alpha in ALPHAS:
                alpha_key = str(alpha)
                for variant in VARIANTS:
                    evaluated.append(
                        {
                            "dataset": dataset,
                            "candidate_index": index,
                            "variant": variant,
                            "alpha": float(alpha),
                            "rank": _rank_for(candidate, variant, alpha_key),
                            "exp292_tier": int(candidate["exp292_tier"]),
                            "grammar_support_count": int(
                                candidate["grammar_support_count"][alpha_key]
                            ),
                            "is_true_anomaly": _is_anomaly(labels[index]),
                            "retrospective": True,
                            "prospective": False,
                            "human_assisted_review_metric": True,
                        }
                    )

    summary: list[dict[str, object]] = []
    dataset_top1: list[dict[str, object]] = []
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in evaluated:
        grouped.setdefault((str(row["variant"]), float(row["alpha"])), []).append(row)

    baseline_precision: dict[tuple[float, str, int], float] = {}
    baseline_ranks: dict[float, dict[tuple[str, int], int]] = {}
    for alpha in ALPHAS:
        rows = grouped[("exp292_baseline", float(alpha))]
        baseline_ranks[float(alpha)] = {
            (str(row["dataset"]), int(row["candidate_index"])): int(row["rank"])
            for row in rows
        }

    for (variant, alpha), rows in sorted(grouped.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row["rank"]),
                str(row["dataset"]),
                int(row["candidate_index"]),
            ),
        )
        total_tp = sum(bool(row["is_true_anomaly"]) for row in ordered)
        for workload in workloads:
            selected = ordered[: int(workload)]
            tp = sum(bool(row["is_true_anomaly"]) for row in selected)
            reviewed = len(selected)
            result: dict[str, object] = {
                "variant": variant,
                "alpha": alpha,
                "workload_type": "global_k",
                "workload_k": int(workload),
                "reviewed": reviewed,
                "tp_at_k": tp,
                "fp_at_k": reviewed - tp,
                "precision_at_k": tp / reviewed if reviewed else 0.0,
                "review_tp_capture_rate": tp / total_tp if total_tp else 0.0,
                "total_standard_review_candidates": len(ordered),
                "total_standard_review_tp": total_tp,
                "total_standard_review_fp": len(ordered) - total_tp,
                "test_metrics_select_policy": False,
                "retrospective": True,
                "prospective": False,
                "human_assisted_review_metric": True,
            }
            summary.append(result)
            if variant == "exp292_baseline":
                baseline_precision[(alpha, "global_k", int(workload))] = float(
                    result["precision_at_k"]
                )

        by_dataset: dict[str, list[dict[str, object]]] = {}
        for row in ordered:
            by_dataset.setdefault(str(row["dataset"]), []).append(row)
        selected_top: list[dict[str, object]] = []
        for dataset in sorted(datasets):
            candidates = sorted(
                by_dataset.get(dataset, []),
                key=lambda row: (
                    int(row["rank"]),
                    int(row["candidate_index"]),
                ),
            )
            selected = candidates[:1]
            selected_top.extend(selected)
            dataset_top1.append(
                {
                    "variant": variant,
                    "alpha": alpha,
                    "dataset": dataset,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected),
                    "selected_index": (
                        int(selected[0]["candidate_index"]) if selected else ""
                    ),
                    "selected_is_true_anomaly": (
                        bool(selected[0]["is_true_anomaly"]) if selected else ""
                    ),
                    "retrospective": True,
                }
            )
        top_tp = sum(bool(row["is_true_anomaly"]) for row in selected_top)
        top_reviewed = len(selected_top)
        top_result: dict[str, object] = {
            "variant": variant,
            "alpha": alpha,
            "workload_type": "per_dataset_top1",
            "workload_k": 1,
            "reviewed": top_reviewed,
            "tp_at_k": top_tp,
            "fp_at_k": top_reviewed - top_tp,
            "precision_at_k": top_tp / top_reviewed if top_reviewed else 0.0,
            "review_tp_capture_rate": top_tp / total_tp if total_tp else 0.0,
            "total_standard_review_candidates": len(ordered),
            "total_standard_review_tp": total_tp,
            "total_standard_review_fp": len(ordered) - total_tp,
            "test_metrics_select_policy": False,
            "retrospective": True,
            "prospective": False,
            "human_assisted_review_metric": True,
        }
        summary.append(top_result)
        if variant == "exp292_baseline":
            baseline_precision[(alpha, "per_dataset_top1", 1)] = float(
                top_result["precision_at_k"]
            )

        comparison_ranks = {
            (str(row["dataset"]), int(row["candidate_index"])): int(row["rank"])
            for row in rows
        }
        baseline = baseline_ranks[alpha]
        rank_delta = {
            key: baseline[key] - comparison_ranks[key]
            for key in baseline
        }
        for result in summary:
            if result["variant"] == variant and result["alpha"] == alpha:
                result["rank_spearman_vs_exp292"] = _spearman(
                    baseline,
                    comparison_ranks,
                )
                result["rank_moved_up"] = sum(value > 0 for value in rank_delta.values())
                result["rank_unchanged"] = sum(value == 0 for value in rank_delta.values())
                result["rank_moved_down"] = sum(value < 0 for value in rank_delta.values())

    for result in summary:
        key = (
            float(result["alpha"]),
            str(result["workload_type"]),
            int(result["workload_k"]),
        )
        base = baseline_precision[key]
        result["baseline_precision_at_k"] = base
        result["precision_lift_vs_exp292"] = float(result["precision_at_k"]) - base
    return evaluated, summary, dataset_top1


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prediction(
    output_root: Path,
    *,
    label_loader: Callable[[set[str]], Mapping[str, Sequence[object]]] = load_test_labels,
) -> dict[str, object]:
    output_root = Path(output_root)
    prediction_rows, prediction_manifest = load_verified_prediction(
        output_root / PREDICTION_NAME,
        output_root / MANIFEST_NAME,
    )
    dataset_names = {str(row["dataset"]) for row in prediction_rows}
    labels = label_loader(dataset_names)
    candidate_rows, summary_rows, dataset_rows = evaluate_rows(
        prediction_rows,
        labels,
    )
    candidate_path = output_root / "exp401_candidate_evaluation.csv"
    summary_path = output_root / "exp401_precision_at_k.csv"
    dataset_path = output_root / "exp401_dataset_top1.csv"
    _write_csv(candidate_path, candidate_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(dataset_path, dataset_rows)
    manifest: dict[str, object] = {
        "experiment_id": EXP401_ID,
        "prediction_sha256": prediction_manifest["prediction_sha256"],
        "labels_loaded_after_prediction_verification": True,
        "datasets": len(dataset_names),
        "evaluated_candidate_rows": len(candidate_rows),
        "summary_rows": len(summary_rows),
        "candidate_evaluation_sha256": sha256_file(candidate_path),
        "precision_at_k_sha256": sha256_file(summary_path),
        "dataset_top1_sha256": sha256_file(dataset_path),
        "autonomous_hard_metric": False,
        "human_assisted_review_metric": True,
        "test_metrics_select_policy": False,
        "retrospective": True,
        "prospective": False,
        "deployment_validated": False,
    }
    (output_root / "exp401_evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
