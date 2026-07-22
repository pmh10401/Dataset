from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.diagnostics import load_test_truth

from .freeze import verify_confidence_prediction


VARIANTS = {
    "current_recovery_shadow": None,
    "guard_all_deep": "guard_all_deep",
    "guard_median_deep": "guard_median_deep",
    "guard_consistent_deep": "guard_consistent_deep",
}


def _metrics(indices: set[int], truth: set[int]) -> dict[str, int | float]:
    tp, fp, fn = len(indices & truth), len(indices - truth), len(truth - indices)
    precision = tp / len(indices) if indices else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(indices), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    found = 0
    total = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            found += 1
            total += found / rank
    return total / positives


def evaluate_confidence_predictions(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: Iterable[str],
    output_root: Path,
    truth_loader: Callable[[tuple[str, ...]], Mapping[str, set[int]]] = load_test_truth,
) -> tuple[Path, Path]:
    expected = tuple(sorted(str(value) for value in expected_datasets))
    rows = verify_confidence_prediction(prediction_path, manifest_path, expected_datasets=expected)
    truth_result = truth_loader(expected)
    truth_by_dataset = truth_result[0] if isinstance(truth_result, tuple) else truth_result
    detail: list[dict[str, object]] = []
    ranking_labels: list[int] = []
    ranking_scores: list[float] = []
    for row in rows:
        dataset = str(row["dataset"])
        truth = set(truth_by_dataset[dataset])
        records = list(row["candidate_records"])
        hard = _metrics(set(row["hard_indices"]), truth)
        for record in records:
            ranking_labels.append(int(record["candidate_index"] in truth))
            ranking_scores.append(float(record["confidence_score"]))
        for variant, field in VARIANTS.items():
            indices = {
                int(record["candidate_index"]) for record in records
                if field is None or bool(record[field])
            }
            candidate = _metrics(indices, truth)
            detail.append({
                "dataset": dataset, "variant": variant,
                "n_train": int(row["n_train"]), "n_test": int(row["n_test"]),
                "true_anomaly_count": len(truth),
                **{f"candidate_{key}": value for key, value in candidate.items()},
                **{f"hard_{key}": value for key, value in hard.items()},
                "hard_policy_changed": False, "human_assisted_only": True,
                "retrospective": True,
            })

    ranking_ap = _average_precision(ranking_labels, ranking_scores)
    summary: list[dict[str, object]] = []
    for variant in VARIANTS:
        selected = [row for row in detail if row["variant"] == variant]
        total = lambda key: sum(int(row[key]) for row in selected)
        count, tp, fp = total("candidate_count"), total("candidate_tp"), total("candidate_fp")
        hard_count, hard_tp, hard_fp = total("hard_count"), total("hard_tp"), total("hard_fp")
        summary.append({
            "experiment_id": "experiment_277_recovery_shadow_confidence_evaluation",
            "variant": variant, "datasets": len(selected),
            "candidate_count": count, "candidate_tp": tp, "candidate_fp": fp,
            "candidate_precision": tp / count if count else 0.0,
            "candidate_mean_f1": sum(float(row["candidate_f1"]) for row in selected) / len(selected),
            "hard_count": hard_count, "hard_tp": hard_tp, "hard_fp": hard_fp,
            "hard_precision": hard_tp / hard_count if hard_count else 0.0,
            "mean_hard_f1": sum(float(row["hard_f1"]) for row in selected) / len(selected),
            "confidence_ranking_average_precision": ranking_ap,
            "hard_policy_changed": False, "human_assisted_only": True,
            "retrospective": True, "prospective": False,
        })

    root = Path(output_root); root.mkdir(parents=True, exist_ok=True)
    detail_path, summary_path = root / "exp277_dataset_results.csv", root / "exp277_summary.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0])); writer.writeheader(); writer.writerows(detail)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (root / "exp277_evaluation_manifest.json").write_text(json.dumps({
        "prediction_sha256": json.loads(Path(manifest_path).read_text())["prediction_sha256"],
        "labels_read_after_prediction_verification": True,
        "datasets": len(expected), "rows": len(detail),
        "hard_policy_changed": False, "human_assisted_only": True,
        "retrospective": True, "prospective": False,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path


__all__ = ["evaluate_confidence_predictions"]
