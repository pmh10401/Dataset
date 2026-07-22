from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.exp309_316_fn_recovery.config import OUTPUT_ROOTS as FN_OUTPUT_ROOTS
from experiments.exp309_316_fn_recovery.contracts import verify_prediction_rows
from experiments.exp309_316_fn_recovery.reporting import write_csv, write_result_report

from .config import ALPHAS, EXP291_MANIFEST, EXP291_PREDICTION, EXPERIMENT_IDS, SHARE_THRESHOLDS
from .contracts import sha256_file, verify_frozen_jsonl


def evaluate_candidate_set(
    *, candidates: set[int], truth: set[int], hard: set[int], existing_review: set[int],
    exp316_candidates: set[int],
) -> dict[str, object]:
    tp = len(candidates & truth)
    fp = len(candidates - truth)
    fn = len(truth - candidates)
    overlap = len(candidates & exp316_candidates)
    union = len(candidates | exp316_candidates)
    return {
        "candidate_count": len(candidates),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(candidates) if candidates else 0.0,
        "candidate_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "outside_hard_tp": len((candidates - hard) & truth),
        "outside_all_existing_tp": len((candidates - hard - existing_review) & truth),
        "zero_hard_dataset_recovered": not bool(hard & truth) and bool(candidates & truth),
        "exp316_overlap_count": overlap,
        "exp316_exact_jaccard": overlap / union if union else 1.0,
    }


def _variant_fields() -> list[tuple[str, str, float, float]]:
    output = []
    for alpha in ALPHAS:
        alpha_key = str(alpha).replace(".", "p")
        for share in SHARE_THRESHOLDS:
            share_key = f"{share:.2f}".replace(".", "p")
            output.append((
                f"alpha={alpha}:share={share:.2f}",
                f"alpha_{alpha_key}_share_{share_key}_indices",
                alpha,
                share,
            ))
    return output


def _distribution(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "candidate_mean": float(np.mean(array)),
        "candidate_median": float(np.median(array)),
        "candidate_p90": float(np.quantile(array, 0.90)),
        "candidate_p95": float(np.quantile(array, 0.95)),
        "candidate_max": float(np.max(array)),
    }


def evaluate_exp318(root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    root = Path(root)
    prediction = root / "exp318_frozen_predictions.jsonl"
    manifest = root / "exp318_prediction_manifest.json"
    rows = verify_frozen_jsonl(prediction, manifest)
    frozen_prediction_sha256 = sha256_file(prediction)

    baseline_rows = verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    baseline = {str(row["dataset"]): row for row in baseline_rows}
    exp316_root = FN_OUTPUT_ROOTS[316]
    exp316_rows = verify_prediction_rows(
        exp316_root / "exp316_frozen_predictions.jsonl",
        exp316_root / "exp316_prediction_manifest.json",
    )
    exp316 = {str(row["dataset"]): row for row in exp316_rows}

    # Labels are deliberately imported only after the frozen prediction hash is verified.
    from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation

    labels = {
        str(row["dataset"]): np.asarray(
            load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8,
        )
        for row in rows
    }
    detail: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for variant, field, alpha, share in _variant_fields():
        selected_rows = []
        alpha_key = str(alpha).replace(".", "p")
        for row in rows:
            dataset = str(row["dataset"])
            truth = set(np.flatnonzero(labels[dataset] == 1).astype(int).tolist())
            candidates = set(map(int, row[field]))
            if any(index < 0 or index >= len(labels[dataset]) for index in candidates):
                raise ValueError(f"candidate index out of range: {dataset}/{variant}")
            base = baseline[dataset]
            hard = set(map(int, base["automatic_hard_indices"]))
            existing_review = (
                set(map(int, base["default_review_indices"]))
                | set(map(int, base["research_only_recovery_priority_indices"]))
            )
            exp316_candidates = set(map(
                int, exp316[dataset][f"alpha_{alpha_key}_two_group_shadow_guarded_indices"],
            ))
            metrics = evaluate_candidate_set(
                candidates=candidates,
                truth=truth,
                hard=hard,
                existing_review=existing_review,
                exp316_candidates=exp316_candidates,
            )
            combined = hard | candidates
            combined_tp = len(combined & truth)
            combined_fp = len(combined - truth)
            combined_fn = len(truth - combined)
            record = {
                "experiment_id": EXPERIMENT_IDS[318],
                "dataset": dataset,
                "variant": variant,
                "alpha": alpha,
                "share_threshold": share,
                "n_test": len(labels[dataset]),
                "true_anomaly_count": len(truth),
                **metrics,
                "combined_f1": 2 * combined_tp / (2 * combined_tp + combined_fp + combined_fn)
                if (2 * combined_tp + combined_fp + combined_fn) else 0.0,
                "retrospective": True,
                "autonomous": False,
                "human_assisted_only": True,
            }
            detail.append(record)
            selected_rows.append(record)
        count = sum(int(row["candidate_count"]) for row in selected_rows)
        tp = sum(int(row["tp"]) for row in selected_rows)
        fp = sum(int(row["fp"]) for row in selected_rows)
        summary.append({
            "experiment_id": EXPERIMENT_IDS[318],
            "variant": variant,
            "alpha": alpha,
            "share_threshold": share,
            "datasets": len(selected_rows),
            "candidate_count": count,
            "tp": tp,
            "fp": fp,
            "precision": tp / count if count else 0.0,
            "mean_candidate_f1": float(np.mean([float(row["candidate_f1"]) for row in selected_rows])),
            "mean_combined_f1": float(np.mean([float(row["combined_f1"]) for row in selected_rows])),
            "outside_hard_tp": sum(int(row["outside_hard_tp"]) for row in selected_rows),
            "outside_all_existing_tp": sum(int(row["outside_all_existing_tp"]) for row in selected_rows),
            "zero_hard_datasets_recovered": sum(bool(row["zero_hard_dataset_recovered"]) for row in selected_rows),
            "exp316_overlap_count": sum(int(row["exp316_overlap_count"]) for row in selected_rows),
            **_distribution([int(row["candidate_count"]) for row in selected_rows]),
            "automatic_policy_changed": False,
            "human_assisted_only": True,
            "primary_policy_selected": False,
            "retrospective": True,
        })
    write_csv(root / "exp318_results.csv", detail)
    write_csv(root / "exp318_summary.csv", summary)
    write_result_report(root / "RESULT_REPORT.md", EXPERIMENT_IDS[318], summary)
    (root / "exp318_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_IDS[318],
        "prediction_sha256": frozen_prediction_sha256,
        "labels_loaded_after_prediction_freeze": True,
        "variants": [row["variant"] for row in summary],
        "automatic_policy_changed": False,
        "human_assisted_only": True,
        "primary_policy_selected": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return detail, summary
