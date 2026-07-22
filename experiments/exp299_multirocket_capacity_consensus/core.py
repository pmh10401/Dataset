from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EXPERIMENT_ID = "experiment_299_multirocket_capacity_consensus"
PRIMARY_ALPHA = 0.005
ALPHAS = (0.005, 0.01, 0.02, 0.05)


def _indices(row: Mapping[str, object], field: str) -> set[int]:
    return {int(value) for value in row.get(field, [])}


def _metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _stratum(n_train: int) -> str:
    return "100-199" if n_train < 200 else "200+"


def build_consensus_row(
    block_512: Mapping[str, object], block_1024: Mapping[str, object], baseline: Mapping[str, object]
) -> dict[str, object]:
    datasets = {str(block_512["dataset"]), str(block_1024["dataset"]), str(baseline["dataset"])}
    if len(datasets) != 1:
        raise ValueError(f"dataset mismatch: {sorted(datasets)}")
    dataset = str(baseline["dataset"])
    alpha_512 = float(block_512["alpha"])
    alpha_1024 = float(block_1024["alpha"])
    if alpha_512 != alpha_1024:
        raise ValueError(f"alpha mismatch: {alpha_512} != {alpha_1024}")
    n_test = int(baseline["n_test"])
    if int(block_512["n_test"]) != n_test or int(block_1024["n_test"]) != n_test:
        raise ValueError(f"n_test mismatch: {dataset}")
    candidates_512 = _indices(block_512, "candidate_indices")
    candidates_1024 = _indices(block_1024, "candidate_indices")
    for values in (candidates_512, candidates_1024):
        if values and (min(values) < 0 or max(values) >= n_test):
            raise ValueError(f"candidate outside TEST range: {dataset}")
    hard = _indices(baseline, "current_hard_indices")
    consensus = candidates_512 & candidates_1024
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": dataset,
        "n_train": int(baseline["n_train"]),
        "n_test": n_test,
        "size_stratum": _stratum(int(baseline["n_train"])),
        "alpha": alpha_512,
        "primary_alpha": alpha_512 == PRIMARY_ALPHA,
        "block_512_variant": str(block_512["variant"]),
        "block_1024_variant": str(block_1024["variant"]),
        "block_512_candidate_indices": sorted(candidates_512),
        "block_1024_candidate_indices": sorted(candidates_1024),
        "capacity_consensus_indices": sorted(consensus),
        "current_hard_indices": sorted(hard),
        "residual_block_512_indices": sorted(candidates_512 - hard),
        "residual_block_1024_indices": sorted(candidates_1024 - hard),
        "residual_consensus_indices": sorted(consensus - hard),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_policy": False,
        "autonomous_policy_changed": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }


def evaluate_predictions(
    predictions: list[dict[str, object]], labels_by_dataset: Mapping[str, np.ndarray]
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for prediction in predictions:
        dataset = str(prediction["dataset"])
        labels = np.asarray(labels_by_dataset[dataset], dtype=np.int8).reshape(-1)
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch: {dataset}")
        truth = set(np.flatnonzero(labels == 1).astype(int))
        block_512 = _indices(prediction, "residual_block_512_indices")
        block_1024 = _indices(prediction, "residual_block_1024_indices")
        consensus = _indices(prediction, "residual_consensus_indices")
        baseline = _indices(prediction, "current_hard_indices")
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for prefix, selected in (
            ("block_512_residual", block_512),
            ("block_1024_residual", block_1024),
            ("consensus_residual", consensus),
            ("baseline_hard", baseline),
            ("counterfactual", baseline | consensus),
        ):
            row.update({f"{prefix}_{key}": value for key, value in _metrics(selected, truth).items()})
        row["zero_f1_before"] = int(float(row["baseline_hard_f1"]) == 0.0)
        row["zero_f1_after"] = int(float(row["counterfactual_f1"]) == 0.0)
        row["zero_f1_fixed"] = int(row["zero_f1_before"] == 1 and row["zero_f1_after"] == 0)
        row["zero_f1_new"] = int(row["zero_f1_before"] == 0 and row["zero_f1_after"] == 1)
        evaluated.append(row)
    return evaluated


def summarize_evaluated(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for alpha in ALPHAS:
        alpha_rows = [row for row in evaluated if float(row["alpha"]) == alpha]
        for stratum in ("100-199", "200+", "100+"):
            rows = alpha_rows if stratum == "100+" else [row for row in alpha_rows if row["size_stratum"] == stratum]
            if not rows:
                continue
            summary: dict[str, object] = {
                "experiment_id": EXPERIMENT_ID,
                "alpha": alpha,
                "primary_alpha": alpha == PRIMARY_ALPHA,
                "size_stratum": stratum,
                "datasets": len(rows),
                "active_datasets": sum(int(row["consensus_residual_count"]) > 0 for row in rows),
            }
            for prefix in ("block_512_residual", "block_1024_residual", "consensus_residual", "baseline_hard", "counterfactual"):
                count = sum(int(row[f"{prefix}_count"]) for row in rows)
                tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
                fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
                summary.update({
                    f"{prefix}_count": count,
                    f"{prefix}_tp": tp,
                    f"{prefix}_fp": fp,
                    f"{prefix}_precision": tp / count if count else 0.0,
                    f"{prefix}_mean_f1": float(np.mean([float(row[f"{prefix}_f1"]) for row in rows])),
                })
            summary.update({
                "retrospective_80pct_point_gate": int(
                    int(summary["consensus_residual_count"]) > 0
                    and float(summary["consensus_residual_precision"]) >= 0.80
                ),
                "zero_f1_before": sum(int(row["zero_f1_before"]) for row in rows),
                "zero_f1_after": sum(int(row["zero_f1_after"]) for row in rows),
                "zero_f1_fixed": sum(int(row["zero_f1_fixed"]) for row in rows),
                "zero_f1_new": sum(int(row["zero_f1_new"]) for row in rows),
                "autonomous_policy_changed": False,
                "retrospective_counterfactual": True,
                "prospective_validated": False,
            })
            summaries.append(summary)
    return summaries
