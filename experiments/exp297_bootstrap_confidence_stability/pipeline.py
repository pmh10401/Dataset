from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS, PRIMARY_EXPERTS
from experiments.exp296_cross_gate_hard_confidence_consensus.routing import CONSENSUS_TIERS

from .core import bootstrap_alert_stability


def _support_map(gate_rows: Sequence[Mapping[str, object]], dataset: str) -> dict[int, set[str]]:
    if len(gate_rows) != len(GATE_VARIANTS) or {str(row["gate_variant"]) for row in gate_rows} != set(GATE_VARIANTS):
        raise ValueError(f"Exp295 Gate coverage mismatch: {dataset}")
    maps = []
    for row in gate_rows:
        maps.append({
            int(record["candidate_index"]): set(map(str, record.get("supporting_experts", [])))
            for record in row["hard_confidence_records"]
        })
    if any(item != maps[0] for item in maps[1:]):
        raise ValueError(f"Exp295 supporting expert mismatch: {dataset}")
    return maps[0]


def build_prediction_rows(
    *,
    exp296_rows: Sequence[Mapping[str, object]],
    exp295_rows: Sequence[Mapping[str, object]],
    primary_train_scores: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    expected_datasets: int,
    replicates: int,
    seed: int,
    alpha: float,
    progress_callback: Callable[[str, int], None] | None = None,
) -> list[dict[str, object]]:
    if len(exp296_rows) != expected_datasets:
        raise ValueError("Exp296 dataset coverage mismatch")
    if set(primary_train_scores) != set(PRIMARY_EXPERTS):
        raise ValueError("primary TRAIN expert coverage mismatch")
    gates_by_dataset: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in exp295_rows:
        gates_by_dataset[str(row["dataset"])].append(row)
    if len(gates_by_dataset) != expected_datasets:
        raise ValueError("Exp295 dataset coverage mismatch")

    output: list[dict[str, object]] = []
    for done, baseline in enumerate(sorted(exp296_rows, key=lambda row: str(row["dataset"])), start=1):
        dataset = str(baseline["dataset"])
        baseline_records = list(baseline["consensus_records"])
        baseline_tiers = {
            int(record["candidate_index"]): str(record["consensus_tier"])
            for record in baseline_records
        }
        support = _support_map(gates_by_dataset[dataset], dataset)
        if set(support) != set(baseline_tiers):
            raise ValueError(f"Exp295/Exp296 Hard mismatch: {dataset}")
        train_by_expert = {
            expert: primary_train_scores[expert].get(dataset, {})
            for expert in PRIMARY_EXPERTS
        }
        stability = bootstrap_alert_stability(
            dataset=dataset,
            baseline_tiers=baseline_tiers,
            supporting_experts_by_index=support,
            train_scores_by_expert=train_by_expert,
            replicates=replicates,
            seed=seed,
            alpha=alpha,
        )
        records = [
            {
                **dict(record),
                **stability[int(record["candidate_index"])],
            }
            for record in baseline_records
        ]
        output.append({
            "experiment_id": "experiment_297_bootstrap_confidence_stability",
            "dataset": dataset,
            "n_test": int(baseline["n_test"]),
            "automatic_hard_indices": list(map(int, baseline["automatic_hard_indices"])),
            "stability_records": records,
            "bootstrap_replicates": int(replicates),
            "bootstrap_seed": int(seed),
            "automatic_policy_changed": False,
            "hard_candidate_set_changed": False,
            "review_candidate_promoted": False,
            "confidence_annotation_only": True,
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "family_policy_used_by_stability": False,
            "test_length_policy_used_by_stability": False,
        })
        if progress_callback is not None:
            progress_callback(dataset, done)
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
            rates: list[float] = []
            eligible = 0
            records_count = 0
            dataset_f1: list[float] = []
            for row in scoped:
                labels = np.asarray(labels_by_dataset[str(row["dataset"])])
                records = list(row["stability_records"])
                if tier != "all_hard":
                    records = [record for record in records if record["consensus_tier"] == tier]
                local_tp = sum(int(labels[int(record["candidate_index"])] == 1) for record in records)
                local_fp = len(records) - local_tp
                local_truth = int(np.count_nonzero(labels == 1))
                dataset_f1.append(float(_metrics(local_tp, local_fp, local_truth)["hard_micro_f1"]))
                tp += local_tp
                fp += local_fp
                records_count += len(records)
                for record in records:
                    if bool(record["bootstrap_eligible"]):
                        eligible += 1
                        rates.append(float(record["stability_rate"]))
            output.append({
                "experiment_id": "experiment_297_bootstrap_confidence_stability",
                "scope": scope,
                "consensus_tier": tier,
                "datasets": len(scoped),
                "truth_anomalies": truth,
                **_metrics(tp, fp, truth),
                "mean_hard_f1": float(np.mean(dataset_f1)) if dataset_f1 else 0.0,
                "eligible_alerts": eligible,
                "abstained_alerts": records_count - eligible,
                "median_stability_rate": float(np.median(rates)) if rates else None,
                "mean_stability_rate": float(np.mean(rates)) if rates else None,
                "automatic_policy_changed": False,
                "confidence_annotation_only": True,
                "retrospective": True,
                "prospective_validated": False,
                "operational_policy_selected": False,
            })
    return output
