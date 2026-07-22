from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.diagnostics import load_test_truth

from .freeze import verify_prediction
from .policy import VARIANT_FIELDS


def _metrics(indices: set[int], truth: set[int]) -> dict[str, int | float]:
    tp, fp, fn = len(indices & truth), len(indices - truth), len(truth - indices)
    precision = tp / len(indices) if indices else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(indices), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_datasets: Iterable[str],
    output_root: Path,
    truth_loader: Callable[[tuple[str, ...]], Mapping[str, set[int]]] = load_test_truth,
) -> tuple[Path, Path]:
    expected = tuple(sorted(str(value) for value in expected_datasets))
    rows = verify_prediction(prediction_path, manifest_path, expected_datasets=expected)
    truth_result = truth_loader(expected)
    truth_by_dataset = truth_result[0] if isinstance(truth_result, tuple) else truth_result
    if set(truth_by_dataset) != set(expected):
        raise ValueError("TEST truth population mismatch")

    detail: list[dict[str, object]] = []
    for row in rows:
        dataset = str(row["dataset"])
        truth = set(truth_by_dataset[dataset])
        hard_metrics = _metrics(set(row["hard_indices"]), truth)
        for variant, field in VARIANT_FIELDS.items():
            candidate = set(row[field])
            metrics = _metrics(candidate, truth)
            detail.append({
                "dataset": dataset, "variant": variant,
                "n_train": int(row["n_train"]), "n_test": int(row["n_test"]),
                "large_train_eligible": bool(row["large_train_eligible"]),
                "true_anomaly_count": len(truth),
                **{f"candidate_{key}": value for key, value in metrics.items()},
                **{f"hard_{key}": value for key, value in hard_metrics.items()},
                "hard_policy_changed": False, "human_assisted_only": True,
                "retrospective": True,
            })

    summary: list[dict[str, object]] = []
    for variant in VARIANT_FIELDS:
        selected = [row for row in detail if row["variant"] == variant]
        total = lambda key: sum(int(row[key]) for row in selected)
        count, tp, fp = total("candidate_count"), total("candidate_tp"), total("candidate_fp")
        hard_count, hard_tp, hard_fp = total("hard_count"), total("hard_tp"), total("hard_fp")
        summary.append({
            "experiment_id": "experiment_275_recovery_shadow_precision_guard",
            "variant": variant, "datasets": len(selected),
            "eligible_datasets": sum(bool(row["large_train_eligible"]) for row in selected),
            "candidate_count": count, "candidate_tp": tp, "candidate_fp": fp,
            "candidate_precision": tp / count if count else 0.0,
            "candidate_mean_f1": sum(float(row["candidate_f1"]) for row in selected) / len(selected),
            "hard_count": hard_count, "hard_tp": hard_tp, "hard_fp": hard_fp,
            "hard_precision": hard_tp / hard_count if hard_count else 0.0,
            "mean_hard_f1": sum(float(row["hard_f1"]) for row in selected) / len(selected),
            "hard_policy_changed": False, "human_assisted_only": True,
            "retrospective": True, "prospective": False,
        })

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    detail_path, summary_path = root / "exp275_dataset_results.csv", root / "exp275_summary.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail[0]))
        writer.writeheader(); writer.writerows(detail)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (root / "exp275_evaluation_manifest.json").write_text(json.dumps({
        "prediction_sha256": json.loads(Path(manifest_path).read_text())["prediction_sha256"],
        "labels_read_after_prediction_verification": True,
        "datasets": len(expected), "rows": len(detail),
        "hard_policy_changed": False, "retrospective": True, "prospective": False,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path


__all__ = ["evaluate_predictions"]
