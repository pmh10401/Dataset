from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def _metrics(indices: Iterable[int], labels: np.ndarray) -> dict[str, float | int]:
    selected = {int(value) for value in indices}
    if selected and (min(selected) < 0 or max(selected) >= len(labels)):
        raise ValueError("Exp237 candidate index outside TEST range")
    truth = set(np.flatnonzero(np.asarray(labels, dtype=int) == 1).astype(int))
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    denominator = 2 * tp + fp + fn
    return {
        "candidate_count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def evaluate_rows(
    rows: Iterable[dict[str, object]], label_loader: Callable[[str], np.ndarray]
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    labels_by_dataset: dict[str, np.ndarray] = {}
    for row in rows:
        dataset = str(row["dataset"])
        if dataset not in labels_by_dataset:
            labels_by_dataset[dataset] = np.asarray(label_loader(dataset), dtype=np.int8)
        labels = labels_by_dataset[dataset]
        if len(labels) != int(row["n_test"]):
            raise ValueError(f"Exp237 TEST length mismatch: {dataset}")
        metrics = _metrics(row.get("candidate_indices", ()), labels)
        evaluated.append(
            {
                **row,
                "true_anomalies": int(np.count_nonzero(labels == 1)),
                **metrics,
                "metric_scope": "retrospective_research_only",
                "autonomous_alert": False,
            }
        )
    return evaluated


def summarize_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["source"]), float(row["alpha"]))].append(row)
    output: list[dict[str, object]] = []
    for (source, alpha), group in sorted(groups.items()):
        tp = sum(int(row["tp"]) for row in group)
        fp = sum(int(row["fp"]) for row in group)
        fn = sum(int(row["fn"]) for row in group)
        counts = np.asarray([int(row["candidate_count"]) for row in group], dtype=float)
        output.append(
            {
                "experiment_id": "experiment_237_new_score_source_qualification",
                "source": source,
                "alpha": alpha,
                "datasets": len(group),
                "candidates": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "mean_f1": float(np.mean([float(row["f1"]) for row in group])),
                "zero_f1_datasets": sum(float(row["f1"]) == 0.0 for row in group),
                "abstained_datasets": sum(bool(row["calibration_abstained"]) for row in group),
                "candidate_mean": float(np.mean(counts)),
                "candidate_median": float(np.median(counts)),
                "candidate_p90": float(np.quantile(counts, 0.90)),
                "candidate_p95": float(np.quantile(counts, 0.95)),
                "candidate_max": int(np.max(counts)),
                "metric_scope": "retrospective_research_only",
                "autonomous_alert": 0,
                "policy_selected_from_test_metrics": 0,
                "comparison_eligible": 0,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
