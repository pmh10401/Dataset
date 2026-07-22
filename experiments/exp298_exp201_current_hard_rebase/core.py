from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EXPERIMENT_ID = "experiment_298_exp201_current_hard_rebase"


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
    if 100 <= n_train < 200:
        return "100-199"
    if n_train >= 200:
        return "200+"
    raise ValueError(f"Exp298 requires n_train >= 100: {n_train}")


def rebase_candidate_row(exp201: Mapping[str, object], exp292: Mapping[str, object]) -> dict[str, object]:
    dataset = str(exp201["dataset"])
    if dataset != str(exp292["dataset"]):
        raise ValueError(f"dataset mismatch: {dataset} != {exp292['dataset']}")
    n_test = int(exp201["n_test"])
    if n_test != int(exp292["n_test"]):
        raise ValueError(f"n_test mismatch: {dataset}")
    source = _indices(exp201, "candidate_indices")
    if source and (min(source) < 0 or max(source) >= n_test):
        raise ValueError(f"candidate outside TEST range: {dataset}")
    hard = _indices(exp292, "automatic_hard_indices")
    standard = _indices(exp292, "standard_review_indices")
    priority = _indices(exp292, "priority_review_indices")
    shadow = _indices(exp292, "shadow_review_indices")
    research = _indices(exp292, "research_only_recovery_priority_indices")
    current_review = standard | priority | shadow | research
    residual = source - hard
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": dataset,
        "variant": str(exp201["variant"]),
        "n_train": int(exp201["n_train"]),
        "n_test": n_test,
        "size_stratum": _stratum(int(exp201["n_train"])),
        "source_exp201_candidate_indices": sorted(source),
        "current_hard_indices": sorted(hard),
        "already_in_current_hard_indices": sorted(source & hard),
        "residual_candidate_indices": sorted(residual),
        "residual_review_overlap_indices": sorted(residual & current_review),
        "residual_outside_current_lanes_indices": sorted(residual - current_review),
        "source_exp201_candidate_count": len(source),
        "already_in_current_hard_count": len(source & hard),
        "residual_candidate_count": len(residual),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": False,
        "uses_dataset_name_for_policy": False,
        "uses_test_length_for_policy": False,
        "retrospective_counterfactual": True,
        "autonomous_policy_changed": False,
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
        residual = _indices(prediction, "residual_candidate_indices")
        baseline = _indices(prediction, "current_hard_indices")
        combined = baseline | residual
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for prefix, selected in (("residual", residual), ("baseline_hard", baseline), ("counterfactual", combined)):
            row.update({f"{prefix}_{key}": value for key, value in _metrics(selected, truth).items()})
        row["zero_f1_before"] = int(float(row["baseline_hard_f1"]) == 0.0)
        row["zero_f1_after"] = int(float(row["counterfactual_f1"]) == 0.0)
        row["zero_f1_fixed"] = int(row["zero_f1_before"] == 1 and row["zero_f1_after"] == 0)
        row["zero_f1_new"] = int(row["zero_f1_before"] == 0 and row["zero_f1_after"] == 1)
        evaluated.append(row)
    return evaluated


def summarize_evaluated(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    groups = (
        ("100-199", [row for row in evaluated if row["size_stratum"] == "100-199"]),
        ("200+", [row for row in evaluated if row["size_stratum"] == "200+"]),
        ("100+", list(evaluated)),
    )
    for size_stratum, rows in groups:
        if not rows:
            continue
        residual_count = sum(int(row["residual_count"]) for row in rows)
        residual_tp = sum(int(row["residual_tp"]) for row in rows)
        residual_fp = sum(int(row["residual_fp"]) for row in rows)
        baseline_tp = sum(int(row["baseline_hard_tp"]) for row in rows)
        baseline_fp = sum(int(row["baseline_hard_fp"]) for row in rows)
        final_tp = sum(int(row["counterfactual_tp"]) for row in rows)
        final_fp = sum(int(row["counterfactual_fp"]) for row in rows)
        summaries.append({
            "experiment_id": EXPERIMENT_ID,
            "size_stratum": size_stratum,
            "datasets": len(rows),
            "active_datasets": sum(int(row["residual_count"]) > 0 for row in rows),
            "source_exp201_candidates": sum(int(row["source_exp201_candidate_count"]) for row in rows),
            "already_in_current_hard": sum(int(row["already_in_current_hard_count"]) for row in rows),
            "residual_candidates": residual_count,
            "residual_tp": residual_tp,
            "residual_fp": residual_fp,
            "residual_precision": residual_tp / residual_count if residual_count else 0.0,
            "retrospective_80pct_point_gate": int(residual_count > 0 and residual_tp / residual_count >= 0.80),
            "baseline_hard_tp": baseline_tp,
            "baseline_hard_fp": baseline_fp,
            "baseline_hard_precision": baseline_tp / (baseline_tp + baseline_fp) if baseline_tp + baseline_fp else 0.0,
            "baseline_mean_hard_f1": float(np.mean([float(row["baseline_hard_f1"]) for row in rows])),
            "counterfactual_tp": final_tp,
            "counterfactual_fp": final_fp,
            "counterfactual_precision": final_tp / (final_tp + final_fp) if final_tp + final_fp else 0.0,
            "counterfactual_mean_f1": float(np.mean([float(row["counterfactual_f1"]) for row in rows])),
            "zero_f1_before": sum(int(row["zero_f1_before"]) for row in rows),
            "zero_f1_after": sum(int(row["zero_f1_after"]) for row in rows),
            "zero_f1_fixed": sum(int(row["zero_f1_fixed"]) for row in rows),
            "zero_f1_new": sum(int(row["zero_f1_new"]) for row in rows),
            "autonomous_policy_changed": False,
            "retrospective_counterfactual": True,
            "prospective_validated": False,
        })
    return summaries
