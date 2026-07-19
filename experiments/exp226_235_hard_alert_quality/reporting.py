from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np


def _metrics(indices: Iterable[int], labels: np.ndarray) -> dict[str, object]:
    selected = {int(value) for value in indices}
    if selected and (min(selected) < 0 or max(selected) >= len(labels)):
        raise ValueError("prediction index outside TEST range")
    truth = set(np.flatnonzero(np.asarray(labels, dtype=int) == 1).astype(int))
    tp = len(selected & truth)
    fp = len(selected - truth)
    fn = len(truth - selected)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    denominator = 2 * tp + fp + fn
    return {
        "count": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def evaluate_prediction_rows(
    rows: Iterable[dict[str, object]],
    *,
    label_loader: Callable[[str], np.ndarray],
) -> list[dict[str, object]]:
    evaluated: list[dict[str, object]] = []
    for row in rows:
        labels = np.asarray(label_loader(str(row["dataset"])), dtype=np.int8)
        hard = _metrics(row.get("hard_indices", ()), labels)
        review = _metrics(row.get("review_indices", ()), labels)
        research = _metrics(row.get("research_indices", ()), labels)
        evaluated.append(
            {
                **row,
                "n_test": len(labels),
                "true_anomalies": int(np.count_nonzero(labels == 1)),
                **{f"hard_{key}": value for key, value in hard.items()},
                **{f"review_{key}": value for key, value in review.items()},
                **{f"research_{key}": value for key, value in research.items()},
                "hard_metric_scope": "autonomous_hard_alert",
                "review_metric_scope": "human_assisted_diagnostic_only",
                "research_metric_scope": "retrospective_research_only",
            }
        )
    return evaluated


def summarize_evaluated_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment_id"]), str(row["variant"]))].append(row)
    summaries: list[dict[str, object]] = []
    for (experiment, variant), group in sorted(groups.items()):
        hard_tp = sum(int(row["hard_tp"]) for row in group)
        hard_fp = sum(int(row["hard_fp"]) for row in group)
        hard_fn = sum(int(row["hard_fn"]) for row in group)
        review_tp = sum(int(row["review_tp"]) for row in group)
        review_fp = sum(int(row["review_fp"]) for row in group)
        research_tp = sum(int(row["research_tp"]) for row in group)
        research_fp = sum(int(row["research_fp"]) for row in group)
        summaries.append(
            {
                "experiment_id": experiment,
                "variant": variant,
                "datasets": len(group),
                "hard_alerts": hard_tp + hard_fp,
                "hard_tp": hard_tp,
                "hard_fp": hard_fp,
                "hard_fn": hard_fn,
                "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
                "hard_recall": hard_tp / max(1, hard_tp + hard_fn),
                "mean_hard_f1": float(np.mean([float(row["hard_f1"]) for row in group])),
                "hard_zero_f1_datasets": sum(float(row["hard_f1"]) == 0.0 for row in group),
                "review_candidates": review_tp + review_fp,
                "review_tp": review_tp,
                "review_fp": review_fp,
                "review_precision": review_tp / max(1, review_tp + review_fp),
                "research_candidates": research_tp + research_fp,
                "research_tp": research_tp,
                "research_fp": research_fp,
                "research_precision": research_tp / max(1, research_tp + research_fp),
                "hard_metric_scope": "autonomous_hard_alert",
                "review_metric_scope": "human_assisted_diagnostic_only",
                "research_metric_scope": "retrospective_research_only",
                "policy_selected_from_test_metrics": 0,
                "retrospective": 1,
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
