from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS, PRIMARY_EXPERTS


EXPANSION_POLICIES = (
    "verified_4of4",
    "verified_3of4",
    "verified_2of4",
    "verified_1of4",
    "diversity_2of2",
    "diversity_1of2",
    "calibration_2of2",
    "calibration_1of2",
)
ALL_POLICIES = ("baseline_no_expansion", *EXPANSION_POLICIES)


def _validate_pair(pair: Sequence[str], name: str) -> list[str]:
    selected = list(map(str, pair))
    if len(selected) not in (0, 2) or len(selected) != len(set(selected)):
        raise ValueError(f"{name} must select two experts or abstain")
    if any(expert not in PRIMARY_EXPERTS for expert in selected):
        raise ValueError(f"{name} selected an unverified expert")
    return selected


def build_dataset_prediction(
    gate_rows: Sequence[Mapping[str, object]],
    *,
    diversity_pair: Sequence[str],
    calibration_pair: Sequence[str],
) -> dict[str, object]:
    by_gate = {str(row["gate_variant"]): row for row in gate_rows}
    if set(by_gate) != set(GATE_VARIANTS) or len(gate_rows) != len(GATE_VARIANTS):
        raise ValueError("three unique Exp294 Gate rows are required")
    rows = [by_gate[variant] for variant in GATE_VARIANTS]
    datasets = {str(row["dataset"]) for row in rows}
    if len(datasets) != 1:
        raise ValueError("Gate rows must belong to one dataset")
    dataset = next(iter(datasets))
    hard_sets = [set(map(int, row["automatic_hard_indices"])) for row in rows]
    standard_sets = [set(map(int, row["standard_review_indices"])) for row in rows]
    if any(values != hard_sets[0] for values in hard_sets[1:]):
        raise ValueError("Hard candidate mismatch across Gates")
    if any(values != standard_sets[0] for values in standard_sets[1:]):
        raise ValueError("Standard candidate mismatch across Gates")
    if hard_sets[0] & standard_sets[0]:
        raise ValueError("Hard and Standard candidates overlap")

    support_maps: list[dict[int, set[str]]] = []
    for row, expected in zip(rows, standard_sets, strict=True):
        records = {
            int(record["candidate_index"]): set(map(str, record["supporting_experts"])) & set(PRIMARY_EXPERTS)
            for record in row["candidate_rank_records"]
        }
        if set(records) != expected:
            raise ValueError("candidate support coverage mismatch")
        support_maps.append(records)
    for index in standard_sets[0]:
        if any(records[index] != support_maps[0][index] for records in support_maps[1:]):
            raise ValueError("verified expert support mismatch across Gates")

    diversity = _validate_pair(diversity_pair, "diversity Gate")
    calibration = _validate_pair(calibration_pair, "calibration Gate")
    promoted = {policy: [] for policy in EXPANSION_POLICIES}
    candidate_records: list[dict[str, object]] = []
    for index in sorted(standard_sets[0]):
        support = support_maps[0][index]
        verified_count = len(support)
        diversity_count = len(support & set(diversity))
        calibration_count = len(support & set(calibration))
        policies: list[str] = []
        for threshold in (4, 3, 2, 1):
            policy = f"verified_{threshold}of4"
            if verified_count >= threshold:
                promoted[policy].append(index)
                policies.append(policy)
        if len(diversity) == 2 and diversity_count == 2:
            promoted["diversity_2of2"].append(index)
            policies.append("diversity_2of2")
        if len(diversity) == 2 and diversity_count >= 1:
            promoted["diversity_1of2"].append(index)
            policies.append("diversity_1of2")
        if len(calibration) == 2 and calibration_count == 2:
            promoted["calibration_2of2"].append(index)
            policies.append("calibration_2of2")
        if len(calibration) == 2 and calibration_count >= 1:
            promoted["calibration_1of2"].append(index)
            policies.append("calibration_1of2")
        candidate_records.append(
            {
                "dataset": dataset,
                "candidate_index": index,
                "supporting_experts": sorted(support),
                "verified_expert_support_count": verified_count,
                "diversity_gate_support_count": diversity_count,
                "calibration_gate_support_count": calibration_count,
                "eligible_expansion_policies": policies,
            }
        )
    hard = sorted(hard_sets[0])
    return {
        "experiment_id": "experiment_303_expanded_train_only_gates",
        "dataset": dataset,
        "n_test": int(rows[0]["n_test"]),
        "automatic_hard_indices": hard,
        "standard_review_indices": sorted(standard_sets[0]),
        "diversity_selected_experts": diversity,
        "calibration_selected_experts": calibration,
        "gate_abstained": len(diversity) == 0 or len(calibration) == 0,
        "candidate_policy_records": candidate_records,
        "promoted_indices_by_policy": promoted,
        "expanded_hard_indices_by_policy": {
            policy: sorted(set(hard) | set(indices)) for policy, indices in promoted.items()
        },
        "automatic_policy_changed": False,
        "counterfactual_expansion_only": True,
        "primary_policy_selected": False,
        "precision_floor_used": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def build_prediction_rows(
    source_rows: Sequence[Mapping[str, object]],
    *,
    diversity_pairs: Mapping[str, Sequence[str]],
    calibration_pairs: Mapping[str, Sequence[str]],
    expected_datasets: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in source_rows:
        grouped[str(row["dataset"])].append(row)
    if len(grouped) != expected_datasets:
        raise ValueError("source dataset coverage mismatch")
    return [
        build_dataset_prediction(
            grouped[dataset],
            diversity_pair=diversity_pairs.get(dataset, ()),
            calibration_pair=calibration_pairs.get(dataset, ()),
        )
        for dataset in sorted(grouped)
    ]


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
            raise ValueError(f"unknown scope: {scope}")
        truth_total = sum(int(np.count_nonzero(labels_by_dataset[str(row["dataset"])] == 1)) for row in scoped)
        baseline_f1: dict[str, float] = {}
        for row in scoped:
            dataset = str(row["dataset"])
            tp, fp = _counts(row["automatic_hard_indices"], labels_by_dataset[dataset])
            truth = int(np.count_nonzero(labels_by_dataset[dataset] == 1))
            baseline_f1[dataset] = _metrics(tp, fp, truth)[2]
        for policy in ALL_POLICIES:
            tp_total = fp_total = promoted_tp = promoted_fp = active = zero_after = zero_fixed = 0
            f1_values: list[float] = []
            workloads: list[int] = []
            for row in scoped:
                dataset = str(row["dataset"])
                labels = labels_by_dataset[dataset]
                promoted = [] if policy == "baseline_no_expansion" else row["promoted_indices_by_policy"][policy]
                automatic = row["automatic_hard_indices"] if policy == "baseline_no_expansion" else row["expanded_hard_indices_by_policy"][policy]
                local_tp, local_fp = _counts(automatic, labels)
                add_tp, add_fp = _counts(promoted, labels)
                truth = int(np.count_nonzero(labels == 1))
                local_f1 = _metrics(local_tp, local_fp, truth)[2]
                active += int(bool(promoted))
                zero_after += int(local_f1 == 0.0)
                zero_fixed += int(baseline_f1[dataset] == 0.0 and local_f1 > 0.0)
                tp_total += local_tp
                fp_total += local_fp
                promoted_tp += add_tp
                promoted_fp += add_fp
                f1_values.append(local_f1)
                workloads.append(len(automatic))
            precision, recall, micro_f1 = _metrics(tp_total, fp_total, truth_total)
            promoted_count = promoted_tp + promoted_fp
            output.append(
                {
                    "experiment_id": "experiment_303_expanded_train_only_gates",
                    "scope": scope,
                    "expansion_policy": policy,
                    "datasets": len(scoped),
                    "active_datasets": active,
                    "gate_abstained_datasets": sum(bool(row["gate_abstained"]) for row in scoped),
                    "truth_anomalies": truth_total,
                    "automatic_count": tp_total + fp_total,
                    "automatic_tp": tp_total,
                    "automatic_fp": fp_total,
                    "automatic_precision": precision,
                    "automatic_recall": recall,
                    "automatic_micro_f1": micro_f1,
                    "mean_automatic_f1": float(np.mean(f1_values)) if f1_values else 0.0,
                    "promoted_count": promoted_count,
                    "promoted_tp": promoted_tp,
                    "promoted_fp": promoted_fp,
                    "promoted_precision": promoted_tp / promoted_count if promoted_count else 0.0,
                    "workload_p95": float(np.quantile(workloads, 0.95)) if workloads else 0.0,
                    "workload_max": max(workloads, default=0),
                    "zero_f1_before": sum(value == 0.0 for value in baseline_f1.values()),
                    "zero_f1_after": zero_after,
                    "zero_f1_fixed": zero_fixed,
                    "precision_floor_used": False,
                    "primary_policy_selected": False,
                    "automatic_policy_changed": False,
                    "retrospective_counterfactual": True,
                    "prospective_validated": False,
                }
            )
    return output
