from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation

from .config import ALPHAS, EXP291_MANIFEST, EXP291_PREDICTION, EXPERIMENT_IDS
from .contracts import sha256_file, verify_prediction_rows
from .reporting import write_csv, write_result_report


def _variant_fields(experiment: int) -> list[tuple[str, str]]:
    output = []
    for alpha in ALPHAS:
        key = str(alpha).replace(".", "p")
        if experiment in (309, 310, 311, 313, 314, 315):
            output.append((f"alpha={alpha}", f"alpha_{key}_indices"))
        elif experiment == 312:
            output.append((f"alpha={alpha}:two_group_shadow", f"alpha_{key}_shadow_indices"))
        elif experiment == 316:
            for variant in (
                "two_group_shadow_raw", "two_group_shadow_guarded",
                "three_group_shadow_raw", "three_group_shadow_guarded",
            ):
                output.append((f"alpha={alpha}:{variant}", f"alpha_{key}_{variant}_indices"))
        else:
            raise ValueError(f"unsupported experiment: {experiment}")
    return output


def _distribution(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "candidate_mean": float(np.mean(array)), "candidate_median": float(np.median(array)),
        "candidate_p90": float(np.quantile(array, 0.90)),
        "candidate_p95": float(np.quantile(array, 0.95)), "candidate_max": float(np.max(array)),
    }


def evaluate_experiment(experiment: int, root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction = root / f"exp{experiment}_frozen_predictions.jsonl"
    manifest = root / f"exp{experiment}_prediction_manifest.json"
    rows = verify_prediction_rows(prediction, manifest)
    baseline = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    }
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in rows
    }
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for variant, field in _variant_fields(int(experiment)):
        variant_rows = []
        for row in rows:
            dataset = str(row["dataset"]); values = labels[dataset]
            candidates = set(map(int, row[field])); truth = set(np.flatnonzero(values == 1).astype(int).tolist())
            base = baseline[dataset]
            hard = set(map(int, base["automatic_hard_indices"]))
            existing = hard | set(map(int, base["default_review_indices"])) | set(map(int, base["research_only_recovery_priority_indices"]))
            tp = len(candidates & truth); fp = len(candidates - truth)
            record = {
                "experiment_id": EXPERIMENT_IDS[int(experiment)], "dataset": dataset,
                "variant": variant, "n_test": len(values), "true_anomaly_count": len(truth),
                "candidate_count": len(candidates), "tp": tp, "fp": fp,
                "precision": tp / len(candidates) if candidates else 0.0,
                "outside_hard_tp": len((candidates - hard) & truth),
                "outside_all_existing_tp": len((candidates - existing) & truth),
                "zero_hard_dataset": not bool(hard & truth),
                "zero_hard_dataset_recovered": not bool(hard & truth) and bool(candidates & truth),
                "retrospective": True, "autonomous": False,
            }
            detail.append(record); variant_rows.append(record)
        candidate_count = sum(int(row["candidate_count"]) for row in variant_rows)
        tp = sum(int(row["tp"]) for row in variant_rows); fp = sum(int(row["fp"]) for row in variant_rows)
        summary.append({
            "experiment_id": EXPERIMENT_IDS[int(experiment)], "variant": variant,
            "datasets": len(variant_rows), "candidate_count": candidate_count,
            "tp": tp, "fp": fp, "precision": tp / candidate_count if candidate_count else 0.0,
            "outside_hard_tp": sum(int(row["outside_hard_tp"]) for row in variant_rows),
            "outside_all_existing_tp": sum(int(row["outside_all_existing_tp"]) for row in variant_rows),
            "zero_hard_datasets_recovered": sum(bool(row["zero_hard_dataset_recovered"]) for row in variant_rows),
            **_distribution([int(row["candidate_count"]) for row in variant_rows]),
            "automatic_hard_count": sum(len(baseline[str(row["dataset"])]["automatic_hard_indices"]) for row in rows),
            "automatic_policy_changed": False, "human_assisted_only": True,
            "primary_policy_selected": False, "retrospective": True,
        })
    write_csv(root / f"exp{experiment}_results.csv", detail)
    write_csv(root / f"exp{experiment}_summary.csv", summary)
    write_result_report(root / "RESULT_REPORT.md", EXPERIMENT_IDS[int(experiment)], summary)
    (root / f"exp{experiment}_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_IDS[int(experiment)],
        "prediction_sha256": sha256_file(prediction),
        "labels_loaded_after_prediction_freeze": True,
        "variants": [row["variant"] for row in summary],
        "primary_policy_selected": False, "automatic_policy_changed": False,
        "retrospective": True, "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return detail, summary
