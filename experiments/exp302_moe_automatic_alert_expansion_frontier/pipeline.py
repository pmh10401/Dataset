from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from .routing import EXPANSION_POLICIES, build_dataset_expansion


ALL_POLICIES = ("baseline_no_expansion", *EXPANSION_POLICIES)


def build_prediction_rows(
    source_rows: Sequence[Mapping[str, object]], *, expected_datasets: int
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["dataset"])].append(row)
    if len(grouped) != expected_datasets:
        raise ValueError(f"Exp294 dataset coverage mismatch: {len(grouped)}/{expected_datasets}")
    predictions: list[dict[str, object]] = []
    for dataset in sorted(grouped):
        row = build_dataset_expansion(grouped[dataset])
        row.update(
            {
                "experiment_id": "experiment_302_moe_automatic_alert_expansion_frontier",
                "automatic_policy_changed": False,
                "counterfactual_expansion_only": True,
                "primary_policy_selected": False,
                "global_top_n_used": False,
                "test_length_policy_used": False,
                "family_policy_used": False,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        )
        predictions.append(row)
    return predictions


def _counts(indices: Sequence[int], labels: np.ndarray) -> tuple[int, int]:
    tp = sum(int(labels[int(index)] == 1) for index in indices)
    return tp, len(indices) - tp


def _metrics(tp: int, fp: int, truth: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_prediction_rows(
    predictions: Sequence[Mapping[str, object]],
    *,
    labels_by_dataset: Mapping[str, np.ndarray],
    scopes: Sequence[str] = ("all_1117", "n_test_ge_100_diagnostic"),
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for scope in scopes:
        if scope == "all_1117":
            scoped = list(predictions)
        elif scope == "n_test_ge_100_diagnostic":
            scoped = [row for row in predictions if int(row["n_test"]) >= 100]
        else:
            raise ValueError(f"unknown evaluation scope: {scope}")
        truth_total = sum(
            int(np.count_nonzero(np.asarray(labels_by_dataset[str(row["dataset"])]) == 1))
            for row in scoped
        )
        baseline_f1_by_dataset: dict[str, float] = {}
        for row in scoped:
            dataset = str(row["dataset"])
            labels = np.asarray(labels_by_dataset[dataset])
            tp, fp = _counts(row["automatic_hard_indices"], labels)
            truth = int(np.count_nonzero(labels == 1))
            baseline_f1_by_dataset[dataset] = _metrics(tp, fp, truth)[2]

        for policy in ALL_POLICIES:
            total_tp = total_fp = promoted_tp = promoted_fp = 0
            f1_values: list[float] = []
            workloads: list[int] = []
            zero_after = zero_fixed = zero_new = 0
            active_datasets = 0
            for row in scoped:
                dataset = str(row["dataset"])
                labels = np.asarray(labels_by_dataset[dataset])
                if policy == "baseline_no_expansion":
                    promoted: list[int] = []
                    automatic = list(map(int, row["automatic_hard_indices"]))
                else:
                    promoted = list(map(int, row["promoted_indices_by_policy"][policy]))
                    automatic = list(map(int, row["expanded_hard_indices_by_policy"][policy]))
                active_datasets += int(bool(promoted))
                local_tp, local_fp = _counts(automatic, labels)
                local_promoted_tp, local_promoted_fp = _counts(promoted, labels)
                truth = int(np.count_nonzero(labels == 1))
                local_f1 = _metrics(local_tp, local_fp, truth)[2]
                baseline_f1 = baseline_f1_by_dataset[dataset]
                zero_after += int(local_f1 == 0.0)
                zero_fixed += int(baseline_f1 == 0.0 and local_f1 > 0.0)
                zero_new += int(baseline_f1 > 0.0 and local_f1 == 0.0)
                total_tp += local_tp
                total_fp += local_fp
                promoted_tp += local_promoted_tp
                promoted_fp += local_promoted_fp
                f1_values.append(local_f1)
                workloads.append(len(automatic))
            precision, recall, micro_f1 = _metrics(total_tp, total_fp, truth_total)
            promoted_count = promoted_tp + promoted_fp
            promoted_precision = promoted_tp / promoted_count if promoted_count else 0.0
            output.append(
                {
                    "experiment_id": "experiment_302_moe_automatic_alert_expansion_frontier",
                    "scope": scope,
                    "expansion_policy": policy,
                    "datasets": len(scoped),
                    "active_datasets": active_datasets,
                    "truth_anomalies": truth_total,
                    "automatic_count": total_tp + total_fp,
                    "automatic_tp": total_tp,
                    "automatic_fp": total_fp,
                    "automatic_precision": precision,
                    "automatic_recall": recall,
                    "automatic_micro_f1": micro_f1,
                    "mean_automatic_f1": float(np.mean(f1_values)) if f1_values else 0.0,
                    "promoted_count": promoted_count,
                    "promoted_tp": promoted_tp,
                    "promoted_fp": promoted_fp,
                    "promoted_precision": promoted_precision,
                    "workload_mean": float(np.mean(workloads)) if workloads else 0.0,
                    "workload_median": float(np.median(workloads)) if workloads else 0.0,
                    "workload_p90": float(np.quantile(workloads, 0.90)) if workloads else 0.0,
                    "workload_p95": float(np.quantile(workloads, 0.95)) if workloads else 0.0,
                    "workload_max": max(workloads, default=0),
                    "zero_f1_before": sum(value == 0.0 for value in baseline_f1_by_dataset.values()),
                    "zero_f1_after": zero_after,
                    "zero_f1_fixed": zero_fixed,
                    "zero_f1_new": zero_new,
                    "precision_floor_used": False,
                    "primary_policy_selected": False,
                    "automatic_policy_changed": False,
                    "retrospective_counterfactual": True,
                    "prospective_validated": False,
                }
            )
    return output
