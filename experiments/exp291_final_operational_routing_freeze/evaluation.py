from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from experiments.exp290_train_raw_exact_normal_guard.evaluation import metrics


VIEWS = {
    "source_baseline_hard": "source_baseline_hard_indices",
    "automatic_hard": "automatic_hard_indices",
    "automatic_normal_exact": "automatic_normal_exact_indices",
    "default_review": "default_review_indices",
    "research_only_recovery_priority": "research_only_recovery_priority_indices",
}


def evaluate_rows(
    predictions: list[dict[str, object]], labels_by_dataset: Mapping[str, np.ndarray]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for prediction in predictions:
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
        "experiment_id": "experiment_291_final_operational_routing_freeze",
        "datasets": len(evaluated),
        "retrospective_integration": True,
        "prospective_validated": False,
        "prediction_uses_test_labels": False,
        "review_metrics_human_assisted_only": True,
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
    for field in (
        "standard_review_indices",
        "priority_review_indices",
        "shadow_review_indices",
        "recovery_shadow_review_indices",
    ):
        summary[field.removesuffix("_indices") + "_count"] = sum(len(row[field]) for row in predictions)  # type: ignore[arg-type]
    summary.update(
        {
            "hard_alert_count": summary["automatic_hard_count"],
            "hard_tp": summary["automatic_hard_tp"],
            "hard_fp": summary["automatic_hard_fp"],
            "hard_fn": summary["automatic_hard_fn"],
            "hard_alert_precision": summary["automatic_hard_precision"],
            "hard_recall": summary["automatic_hard_recall"],
            "hard_mean_dataset_f1": summary["automatic_hard_mean_f1"],
        }
    )
    return evaluated, summary


__all__ = ["VIEWS", "evaluate_rows"]
