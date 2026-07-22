from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.diagnostics import load_test_truth
from experiments.exp278_281_train_only_recovery.contracts import write_csv

from .prediction import VARIANTS, verify_prediction


def _metrics(indices: set[int], truth: set[int]) -> dict[str, int | float]:
    tp, fp, fn = len(indices & truth), len(indices - truth), len(truth - indices)
    precision = tp / len(indices) if indices else 0.0; recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"count": len(indices), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(
    prediction_path: Path, manifest_path: Path, *, output_root: Path,
    truth_loader: Callable[[tuple[str, ...]], Mapping[str, set[int]]] = load_test_truth,
) -> tuple[Path, Path]:
    rows = verify_prediction(prediction_path, manifest_path)
    datasets = tuple(str(row["dataset"]) for row in rows)
    loaded = truth_loader(datasets); truth_map = loaded[0] if isinstance(loaded, tuple) else loaded
    detail = []
    for row in rows:
        truth = set(truth_map[str(row["dataset"])]); hard = _metrics(set(row["hard_indices"]), truth)
        for variant in VARIANTS:
            indices = {int(record["candidate_index"]) for record in row["candidate_records"] if record["variant_pass"][variant]}
            candidate = _metrics(indices, truth)
            detail.append({
                "dataset": row["dataset"], "variant": variant, "n_train": row["n_train"], "n_test": row["n_test"],
                "true_anomaly_count": len(truth), **{f"candidate_{k}": v for k, v in candidate.items()},
                **{f"hard_{k}": v for k, v in hard.items()}, "hard_policy_changed": False,
                "human_assisted_only": True, "retrospective": True,
            })
    summary = []
    for variant in VARIANTS:
        selected = [row for row in detail if row["variant"] == variant]; total = lambda key: sum(int(row[key]) for row in selected)
        count, tp, fp = total("candidate_count"), total("candidate_tp"), total("candidate_fp")
        hc, ht, hf = total("hard_count"), total("hard_tp"), total("hard_fp")
        summary.append({
            "experiment_id": "experiment_284_prototype_distance_evaluation", "variant": variant,
            "datasets": len(selected), "candidate_count": count, "candidate_tp": tp, "candidate_fp": fp,
            "candidate_precision": tp / count if count else 0.0,
            "candidate_mean_f1": sum(float(row["candidate_f1"]) for row in selected) / len(selected),
            "hard_count": hc, "hard_tp": ht, "hard_fp": hf, "hard_precision": ht / hc if hc else 0.0,
            "mean_hard_f1": sum(float(row["hard_f1"]) for row in selected) / len(selected),
            "hard_policy_changed": False, "human_assisted_only": True, "retrospective": True, "prospective": False,
        })
    root = Path(output_root); detail_path = root / "exp284_dataset_results.csv"; summary_path = root / "exp284_summary.csv"
    write_csv(detail_path, detail); write_csv(summary_path, summary)
    (root / "exp284_evaluation_manifest.json").write_text(json.dumps({
        "prediction_sha256": json.loads(Path(manifest_path).read_text())["prediction_sha256"],
        "labels_read_after_prediction_verification": True, "datasets": len(datasets),
        "hard_policy_changed": False, "human_assisted_only": True, "retrospective": True,
    }, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return detail_path, summary_path


__all__ = ["evaluate_predictions"]
