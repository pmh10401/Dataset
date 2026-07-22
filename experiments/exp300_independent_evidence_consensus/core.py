from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EXPERIMENT_ID = "experiment_300_independent_evidence_consensus"
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


def build_consensus_row(
    multirocket: Mapping[str, object],
    structural: Mapping[str, object],
    change_point: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    rows = (multirocket, structural, change_point)
    datasets = {str(row["dataset"]) for row in (*rows, baseline)}
    if len(datasets) != 1:
        raise ValueError(f"dataset mismatch: {sorted(datasets)}")
    dataset = str(baseline["dataset"])
    alphas = {float(row["alpha"]) for row in rows}
    if len(alphas) != 1:
        raise ValueError(f"alpha mismatch: {sorted(alphas)}")
    alpha = alphas.pop()
    n_test = int(baseline["n_test"])
    if any(int(row["n_test"]) != n_test for row in rows):
        raise ValueError(f"n_test mismatch: {dataset}")
    source_sets = {
        "multirocket": _indices(multirocket, "candidate_indices"),
        "structural": _indices(structural, "candidate_indices"),
        "change_point": _indices(change_point, "candidate_indices"),
    }
    for values in source_sets.values():
        if values and (min(values) < 0 or max(values) >= n_test):
            raise ValueError(f"candidate outside TEST range: {dataset}")
    multi, structure, change = source_sets.values()
    consensus_2of3 = (multi & structure) | (multi & change) | (structure & change)
    consensus_3of3 = multi & structure & change
    hard = _indices(baseline, "current_hard_indices")
    return {
        "experiment_id": EXPERIMENT_ID,
        "dataset": dataset,
        "n_train": int(baseline["n_train"]),
        "n_test": n_test,
        "size_stratum": "100-199" if int(baseline["n_train"]) < 200 else "200+",
        "alpha": alpha,
        "primary_alpha": alpha == PRIMARY_ALPHA,
        "multirocket_variant": str(multirocket["variant"]),
        "structural_variant": str(structural["variant"]),
        "change_point_variant": str(change_point["variant"]),
        "multirocket_indices": sorted(multi),
        "structural_indices": sorted(structure),
        "change_point_indices": sorted(change),
        "consensus_2of3_indices": sorted(consensus_2of3),
        "consensus_3of3_indices": sorted(consensus_3of3),
        "current_hard_indices": sorted(hard),
        "residual_multirocket_indices": sorted(multi - hard),
        "residual_structural_indices": sorted(structure - hard),
        "residual_change_point_indices": sorted(change - hard),
        "residual_2of3_indices": sorted(consensus_2of3 - hard),
        "residual_3of3_indices": sorted(consensus_3of3 - hard),
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
        hard = _indices(prediction, "current_hard_indices")
        views = {
            "residual_multirocket": _indices(prediction, "residual_multirocket_indices"),
            "residual_structural": _indices(prediction, "residual_structural_indices"),
            "residual_change_point": _indices(prediction, "residual_change_point_indices"),
            "residual_2of3": _indices(prediction, "residual_2of3_indices"),
            "residual_3of3": _indices(prediction, "residual_3of3_indices"),
            "baseline_hard": hard,
        }
        views["counterfactual_2of3"] = hard | views["residual_2of3"]
        views["counterfactual_3of3"] = hard | views["residual_3of3"]
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for prefix, selected in views.items():
            row.update({f"{prefix}_{key}": value for key, value in _metrics(selected, truth).items()})
        row["zero_f1_before"] = int(float(row["baseline_hard_f1"]) == 0.0)
        for suffix in ("2of3", "3of3"):
            after = int(float(row[f"counterfactual_{suffix}_f1"]) == 0.0)
            row[f"zero_f1_after_{suffix}"] = after
            row[f"zero_f1_fixed_{suffix}"] = int(row["zero_f1_before"] == 1 and after == 0)
        evaluated.append(row)
    return evaluated


def summarize_evaluated(evaluated: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    metric_views = (
        "residual_multirocket", "residual_structural", "residual_change_point",
        "residual_2of3", "residual_3of3", "baseline_hard", "counterfactual_2of3", "counterfactual_3of3",
    )
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
            }
            for prefix in metric_views:
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
                "gate_2of3_80pct": int(int(summary["residual_2of3_count"]) > 0 and float(summary["residual_2of3_precision"]) >= 0.80),
                "gate_3of3_80pct": int(int(summary["residual_3of3_count"]) > 0 and float(summary["residual_3of3_precision"]) >= 0.80),
                "zero_f1_before": sum(int(row["zero_f1_before"]) for row in rows),
                "zero_f1_after_2of3": sum(int(row["zero_f1_after_2of3"]) for row in rows),
                "zero_f1_fixed_2of3": sum(int(row["zero_f1_fixed_2of3"]) for row in rows),
                "zero_f1_after_3of3": sum(int(row["zero_f1_after_3of3"]) for row in rows),
                "zero_f1_fixed_3of3": sum(int(row["zero_f1_fixed_3of3"]) for row in rows),
                "autonomous_policy_changed": False,
                "retrospective_counterfactual": True,
                "prospective_validated": False,
            })
            summaries.append(summary)
    return summaries
