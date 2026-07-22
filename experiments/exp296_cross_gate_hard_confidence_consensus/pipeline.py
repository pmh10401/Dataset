from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS

from .routing import CONSENSUS_TIERS, build_consensus_records


def build_prediction_rows(
    *,
    exp295_rows: Sequence[Mapping[str, object]],
    expected_datasets: int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in exp295_rows:
        grouped[str(row["dataset"])].append(row)
    if len(grouped) != expected_datasets:
        raise ValueError(f"Exp295 dataset coverage mismatch: {len(grouped)}/{expected_datasets}")
    if len(exp295_rows) != expected_datasets * len(GATE_VARIANTS):
        raise ValueError("Exp295 Gate row coverage mismatch")

    output: list[dict[str, object]] = []
    for dataset in sorted(grouped):
        gate_rows = grouped[dataset]
        records = build_consensus_records(dataset=dataset, gate_rows=gate_rows)
        hard_indices = [int(record["candidate_index"]) for record in records]
        n_test_values = {int(row["n_test"]) for row in gate_rows}
        if len(n_test_values) != 1:
            raise ValueError(f"n_test mismatch between Gates: {dataset}")
        output.append(
            {
                "experiment_id": "experiment_296_cross_gate_hard_confidence_consensus",
                "dataset": dataset,
                "n_test": n_test_values.pop(),
                "automatic_hard_indices": hard_indices,
                "consensus_records": records,
                "automatic_policy_changed": False,
                "hard_candidate_set_changed": False,
                "review_candidate_promoted": False,
                "confidence_annotation_only": True,
                "contains_test_values": False,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "family_policy_used_by_consensus": False,
                "test_length_policy_used_by_consensus": False,
                "primary_gate_selected": False,
                "tier_definition_derived_after_exp295_retrospective_analysis": True,
            }
        )
    return output


def _metrics(tp: int, fp: int, truth: int) -> dict[str, float | int]:
    count = tp + fp
    precision = tp / count if count else 0.0
    recall = tp / truth if truth else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hard_count": count,
        "hard_tp": tp,
        "hard_fp": fp,
        "hard_precision": precision,
        "hard_recall": recall,
        "hard_micro_f1": f1,
    }


def evaluate_prediction_rows(
    *,
    predictions: Sequence[Mapping[str, object]],
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
        truth = sum(
            int(np.count_nonzero(np.asarray(labels_by_dataset[str(row["dataset"])]) == 1))
            for row in scoped
        )
        for tier in ("all_hard", *CONSENSUS_TIERS):
            tp = fp = 0
            dataset_f1: list[float] = []
            for row in scoped:
                dataset = str(row["dataset"])
                labels = np.asarray(labels_by_dataset[dataset])
                records = list(row["consensus_records"])
                if tier != "all_hard":
                    records = [record for record in records if record["consensus_tier"] == tier]
                local_tp = sum(int(labels[int(record["candidate_index"])] == 1) for record in records)
                local_fp = len(records) - local_tp
                local_truth = int(np.count_nonzero(labels == 1))
                local = _metrics(local_tp, local_fp, local_truth)
                dataset_f1.append(float(local["hard_micro_f1"]))
                tp += local_tp
                fp += local_fp
            output.append(
                {
                    "experiment_id": "experiment_296_cross_gate_hard_confidence_consensus",
                    "scope": scope,
                    "consensus_tier": tier,
                    "datasets": len(scoped),
                    "truth_anomalies": truth,
                    **_metrics(tp, fp, truth),
                    "mean_hard_f1": float(np.mean(dataset_f1)) if dataset_f1 else 0.0,
                    "automatic_policy_changed": False,
                    "confidence_annotation_only": True,
                    "retrospective": True,
                    "prospective_validated": False,
                    "operational_policy_selected": False,
                }
            )
    return output
