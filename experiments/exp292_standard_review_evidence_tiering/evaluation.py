from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _metrics(selected: set[int], truth: set[int]) -> dict[str, float | int]:
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(selected), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_review_tiers(
    predictions: list[dict[str, object]], labels_by_dataset: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    views = {
        "automatic_hard": "automatic_hard_indices",
        "standard_review": "standard_review_indices",
        "review_priority_1": "review_priority_1_indices",
        "review_priority_2": "review_priority_2_indices",
        "review_priority_3": "review_priority_3_indices",
    }
    evaluated: list[dict[str, object]] = []
    for prediction in predictions:
        dataset = str(prediction["dataset"])
        labels = np.asarray(labels_by_dataset[dataset], dtype=np.int8).reshape(-1)
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"label length mismatch: {dataset}")
        truth = set(np.flatnonzero(labels == 1).astype(int))
        row = dict(prediction)
        row["true_anomaly_count"] = len(truth)
        for view, field in views.items():
            values = _metrics(set(map(int, prediction.get(field, []))), truth)
            row.update({f"{view}_{key}": value for key, value in values.items()})
        evaluated.append(row)
    summary: dict[str, object] = {
        "experiment_id": "experiment_292_standard_review_evidence_tiering",
        "datasets": len(evaluated),
        "review_metrics_human_assisted_only": True,
        "automatic_policy_changed": False,
        "retrospective": True,
        "prospective_validated": False,
    }
    for view in views:
        tp = sum(int(row[f"{view}_tp"]) for row in evaluated)
        fp = sum(int(row[f"{view}_fp"]) for row in evaluated)
        fn = sum(int(row[f"{view}_fn"]) for row in evaluated)
        summary.update({
            f"{view}_count": tp + fp,
            f"{view}_tp": tp,
            f"{view}_fp": fp,
            f"{view}_fn": fn,
            f"{view}_precision": tp / (tp + fp) if tp + fp else 0.0,
            f"{view}_recall": tp / (tp + fn) if tp + fn else 0.0,
            f"{view}_mean_f1": float(np.mean([float(row[f"{view}_f1"]) for row in evaluated])),
        })
    return evaluated, summary
