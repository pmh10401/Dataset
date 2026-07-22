from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

import numpy as np

from .routing import POLICIES, build_routing


ALL_POLICIES = ("baseline_no_expansion", *POLICIES)


def build_prediction_rows(
    source_rows: Sequence[Mapping[str, object]],
    *,
    wavelet_support_by_dataset: Mapping[str, Set[int]],
    expected_datasets: int,
) -> list[dict[str, object]]:
    if len(source_rows) != expected_datasets or len(wavelet_support_by_dataset) != expected_datasets:
        raise ValueError("common-support coverage mismatch")
    source_by_dataset = {str(row["dataset"]): row for row in source_rows}
    if len(source_by_dataset) != expected_datasets or set(source_by_dataset) != set(wavelet_support_by_dataset):
        raise ValueError("common-support coverage mismatch")

    predictions: list[dict[str, object]] = []
    for dataset in sorted(source_by_dataset):
        source = source_by_dataset[dataset]
        records = source["candidate_policy_records"]
        evidence = {
            int(record["candidate_index"]): {
                "existing_support_count": int(record["existing_support_count"]),
                "reconstruction_support": bool(record["reconstruction_support"]),
            }
            for record in records
        }
        prediction = build_routing(
            dataset=dataset,
            hard_indices=source["automatic_hard_indices"],
            standard_indices=source["standard_review_indices"],
            evidence_by_index=evidence,
            wavelet_support=set(wavelet_support_by_dataset[dataset]),
        )
        prediction.update({
            "experiment_id": "experiment_305_wavelet_reconstruction_cross_confirmation",
            "n_test": int(source["n_test"]),
            "common_support_population": True,
            "wavelet_full_coverage_verified": True,
            "reconstruction_provenance_verified": False,
            "research_only": True,
            "automatic_policy_changed": False,
            "counterfactual_expansion_only": True,
            "primary_policy_selected": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        })
        predictions.append(prediction)
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
) -> list[dict[str, object]]:
    truth_total = sum(int(np.count_nonzero(labels_by_dataset[str(row["dataset"])] == 1)) for row in predictions)
    baseline_f1: dict[str, float] = {}
    for row in predictions:
        dataset = str(row["dataset"])
        labels = labels_by_dataset[dataset]
        tp, fp = _counts(row["automatic_hard_indices"], labels)
        baseline_f1[dataset] = _metrics(tp, fp, int(np.count_nonzero(labels == 1)))[2]

    output: list[dict[str, object]] = []
    for policy in ALL_POLICIES:
        tp_total = fp_total = add_tp = add_fp = active = zero_after = zero_fixed = 0
        f1_values: list[float] = []
        workloads: list[int] = []
        for row in predictions:
            dataset = str(row["dataset"])
            labels = labels_by_dataset[dataset]
            promoted = [] if policy == "baseline_no_expansion" else row["promoted_indices_by_policy"][policy]
            automatic = row["automatic_hard_indices"] if policy == "baseline_no_expansion" else row["expanded_hard_indices_by_policy"][policy]
            local_tp, local_fp = _counts(automatic, labels)
            local_add_tp, local_add_fp = _counts(promoted, labels)
            truth = int(np.count_nonzero(labels == 1))
            local_f1 = _metrics(local_tp, local_fp, truth)[2]
            tp_total += local_tp
            fp_total += local_fp
            add_tp += local_add_tp
            add_fp += local_add_fp
            active += int(bool(promoted))
            zero_after += int(local_f1 == 0.0)
            zero_fixed += int(baseline_f1[dataset] == 0.0 and local_f1 > 0.0)
            f1_values.append(local_f1)
            workloads.append(len(automatic))
        precision, recall, micro_f1 = _metrics(tp_total, fp_total, truth_total)
        promoted_count = add_tp + add_fp
        output.append({
            "experiment_id": "experiment_305_wavelet_reconstruction_cross_confirmation",
            "scope": "exp304_reconstruction_common_support_251",
            "expansion_policy": policy,
            "datasets": len(predictions),
            "active_datasets": active,
            "truth_anomalies": truth_total,
            "automatic_count": tp_total + fp_total,
            "automatic_tp": tp_total,
            "automatic_fp": fp_total,
            "automatic_precision": precision,
            "automatic_recall": recall,
            "automatic_micro_f1": micro_f1,
            "mean_automatic_f1": float(np.mean(f1_values)) if f1_values else 0.0,
            "promoted_count": promoted_count,
            "promoted_tp": add_tp,
            "promoted_fp": add_fp,
            "promoted_precision": add_tp / promoted_count if promoted_count else 0.0,
            "workload_p95": float(np.quantile(workloads, 0.95)) if workloads else 0.0,
            "workload_max": max(workloads, default=0),
            "zero_f1_before": sum(value == 0.0 for value in baseline_f1.values()),
            "zero_f1_after": zero_after,
            "zero_f1_fixed": zero_fixed,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
            "retrospective_counterfactual": True,
            "prospective_validated": False,
        })
    return output
