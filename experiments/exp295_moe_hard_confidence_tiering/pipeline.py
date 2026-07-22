from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import (
    GATE_VARIANTS,
    PRIMARY_EXPERTS,
)

from .routing import build_hard_confidence_records


TIERS = ("high", "medium", "needs_confirmation")


def build_prediction_rows(
    *,
    baseline_rows: Sequence[Mapping[str, object]],
    gate_rows: Sequence[Mapping[str, object]],
    all_support: Mapping[str, Mapping[str, set[int]]],
    expected_datasets: int,
) -> list[dict[str, object]]:
    if len(baseline_rows) != expected_datasets:
        raise ValueError("baseline dataset coverage mismatch")
    gate_map = {
        (str(row["dataset"]), str(row["gate_variant"])): row
        for row in gate_rows
    }
    if len(gate_map) != expected_datasets * len(GATE_VARIANTS):
        raise ValueError("Gate choice coverage mismatch")
    if set(PRIMARY_EXPERTS) - set(all_support):
        raise ValueError("primary source support is incomplete")

    predictions: list[dict[str, object]] = []
    for baseline in sorted(baseline_rows, key=lambda row: str(row["dataset"])):
        dataset = str(baseline["dataset"])
        hard_indices = sorted(set(map(int, baseline["automatic_hard_indices"])))
        dataset_support = {
            expert: set(map(int, all_support[expert].get(dataset, set())))
            for expert in PRIMARY_EXPERTS
        }
        for variant in GATE_VARIANTS:
            gate = gate_map.get((dataset, variant))
            if gate is None:
                raise ValueError(f"missing Gate choice: {dataset}/{variant}")
            selected = list(map(str, gate["selected_experts"]))
            records = build_hard_confidence_records(
                dataset=dataset,
                hard_indices=hard_indices,
                source_support=dataset_support,
                selected_experts=selected,
            )
            predictions.append(
                {
                    "experiment_id": "experiment_295_moe_hard_alert_confidence_tiering",
                    "dataset": dataset,
                    "n_test": int(baseline["n_test"]),
                    "gate_variant": variant,
                    "ordered_experts": list(map(str, gate["ordered_experts"])),
                    "selected_experts": selected,
                    "gate_abstained": len(selected) == 0,
                    "automatic_hard_indices": hard_indices,
                    "hard_confidence_records": records,
                    "automatic_policy_changed": False,
                    "hard_candidate_set_changed": False,
                    "review_candidate_promoted": False,
                    "confidence_annotation_only": True,
                    "contains_test_values": False,
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "family_policy_used_by_confidence_tiering": False,
                    "test_length_policy_used_by_confidence_tiering": False,
                }
            )
    return predictions


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
    for variant in GATE_VARIANTS:
        variant_rows = [row for row in predictions if row["gate_variant"] == variant]
        for scope in scopes:
            if scope == "all_1117":
                scoped = variant_rows
            elif scope == "n_test_ge_100_diagnostic":
                scoped = [row for row in variant_rows if int(row["n_test"]) >= 100]
            else:
                raise ValueError(f"unknown evaluation scope: {scope}")
            truth = sum(
                int(np.count_nonzero(np.asarray(labels_by_dataset[str(row["dataset"])]) == 1))
                for row in scoped
            )
            for tier in ("all_hard", *TIERS):
                tp = fp = 0
                dataset_f1: list[float] = []
                for row in scoped:
                    dataset = str(row["dataset"])
                    labels = np.asarray(labels_by_dataset[dataset])
                    records = list(row["hard_confidence_records"])
                    if tier != "all_hard":
                        records = [record for record in records if record["confidence_tier"] == tier]
                    local_tp = sum(int(labels[int(record["candidate_index"])] == 1) for record in records)
                    local_fp = len(records) - local_tp
                    local_truth = int(np.count_nonzero(labels == 1))
                    local = _metrics(local_tp, local_fp, local_truth)
                    dataset_f1.append(float(local["hard_micro_f1"]))
                    tp += local_tp
                    fp += local_fp
                metrics = _metrics(tp, fp, truth)
                output.append(
                    {
                        "experiment_id": "experiment_295_moe_hard_alert_confidence_tiering",
                        "gate_variant": variant,
                        "scope": scope,
                        "confidence_tier": tier,
                        "datasets": len(scoped),
                        "gate_available_datasets": sum(not bool(row["gate_abstained"]) for row in scoped),
                        "truth_anomalies": truth,
                        **metrics,
                        "mean_hard_f1": float(np.mean(dataset_f1)) if dataset_f1 else 0.0,
                        "automatic_policy_changed": False,
                        "confidence_annotation_only": True,
                        "retrospective": True,
                        "prospective_validated": False,
                        "primary_gate_selected": False,
                    }
                )
    return output
