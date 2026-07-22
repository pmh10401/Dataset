from __future__ import annotations

from collections.abc import Mapping

import numpy as np


VIEWS = {
    "baseline_hard": "baseline_hard_indices",
    "guarded_hard": "guarded_hard_indices",
    "train_raw_exact_normal": "train_raw_exact_normal_indices",
}


def metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_rows(
    prediction_rows: list[dict[str, object]],
    labels_by_dataset: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for prediction in prediction_rows:
        dataset = str(prediction["dataset"])
        labels = np.asarray(labels_by_dataset[dataset], dtype=np.int8).reshape(-1)
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"label length mismatch: dataset={dataset}")
        truth = set(np.flatnonzero(labels == 1).astype(int))
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for view, field in VIEWS.items():
            selected = set(map(int, prediction[field]))  # type: ignore[arg-type]
            row.update({f"{view}_{key}": value for key, value in metrics(selected, truth).items()})
        evaluated.append(row)

    summary: dict[str, object] = {
        "experiment_id": "experiment_290_train_raw_exact_normal_guard",
        "datasets": len(evaluated),
        "retrospective_counterfactual": True,
        "prediction_uses_test_labels": False,
        "match_definition": "sqlite_values_blob_byte_exact",
        "upstream_end_to_end_strict_train_only": False,
    }
    for view in VIEWS:
        tp = sum(int(row[f"{view}_tp"]) for row in evaluated)
        fp = sum(int(row[f"{view}_fp"]) for row in evaluated)
        fn = sum(int(row[f"{view}_fn"]) for row in evaluated)
        summary.update(
            {
                f"{view}_count": tp + fp,
                f"{view}_tp": tp,
                f"{view}_fp": fp,
                f"{view}_fn": fn,
                f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
                f"{view}_recall": tp / (tp + fn) if tp + fn else 0.0,
                f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in evaluated])),
            }
        )
    summary["alert_reduction"] = int(summary["baseline_hard_count"]) - int(summary["guarded_hard_count"])
    summary["fp_reduction"] = int(summary["baseline_hard_fp"]) - int(summary["guarded_hard_fp"])
    summary["tp_reduction"] = int(summary["baseline_hard_tp"]) - int(summary["guarded_hard_tp"])
    summary["precision_change"] = float(summary["guarded_hard_precision"]) - float(summary["baseline_hard_precision"])
    summary["mean_f1_change"] = float(summary["guarded_hard_mean_f1"]) - float(summary["baseline_hard_mean_f1"])
    return evaluated, summary


__all__ = ["VIEWS", "evaluate_rows", "metrics"]
