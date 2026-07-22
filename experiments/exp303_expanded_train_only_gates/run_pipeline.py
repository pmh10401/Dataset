from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.artifacts import load_primary_train_scores
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows
from experiments.exp293_294_train_only_moe_lite.routing import PRIMARY_EXPERTS

from .config import (
    EXPECTED_DATASETS,
    EXPECTED_HARD_ALERTS,
    EXPECTED_STANDARD_CANDIDATES,
    EXP293_AUDIT_PATH,
    EXP293_AUDIT_SHA256,
    EXP294_MANIFEST_PATH,
    EXP294_PREDICTION_PATH,
    EXP294_PREDICTION_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
from .gates import choose_calibration_pair, choose_diversity_pair
from .pipeline import ALL_POLICIES, build_prediction_rows, evaluate_prediction_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_audits() -> dict[str, dict[str, dict[str, object]]]:
    if sha256_file(EXP293_AUDIT_PATH) != EXP293_AUDIT_SHA256:
        raise ValueError("Exp293 expert audit SHA256 mismatch")
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    with EXP293_AUDIT_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expert = str(row["expert"])
            if expert not in PRIMARY_EXPERTS:
                continue
            if row["contains_test_values"] != "False" or row["contains_test_labels"] != "False" or row["contains_evaluation_metrics"] != "False":
                raise ValueError("Exp293 audit contains forbidden TEST information")
            grouped[str(row["dataset"])][expert] = {
                "available": row["available"] == "True",
                "provenance_verified": row["provenance_verified"] == "True",
                "risk_upper_95": float(row["risk_upper_95"]),
                "attainable_p": float(row["attainable_p"]),
                "train_drift": float(row["train_drift"]),
                "stability": float(row["stability"]),
            }
    if len(grouped) != EXPECTED_DATASETS:
        raise ValueError("Exp293 audit dataset coverage mismatch")
    return grouped


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(EXP294_PREDICTION_PATH) != EXP294_PREDICTION_SHA256:
        raise ValueError("Exp294 frozen prediction SHA256 mismatch")
    source_rows = verify_rows(EXP294_PREDICTION_PATH, EXP294_MANIFEST_PATH)
    audits = _load_audits()
    primary_scores, source_manifests = load_primary_train_scores()
    datasets = sorted(audits)
    diversity_pairs: dict[str, list[str]] = {}
    calibration_pairs: dict[str, list[str]] = {}
    gate_rows: list[dict[str, object]] = []
    for dataset in datasets:
        train_by_expert = {
            expert: primary_scores[expert][dataset]
            for expert in PRIMARY_EXPERTS
            if dataset in primary_scores.get(expert, {})
        }
        diversity = choose_diversity_pair(train_by_expert, audits[dataset])
        calibration = choose_calibration_pair(audits[dataset])
        diversity_pairs[dataset] = diversity
        calibration_pairs[dataset] = calibration
        gate_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset": dataset,
                "diversity_selected_experts": json.dumps(diversity, separators=(",", ":")),
                "calibration_selected_experts": json.dumps(calibration, separators=(",", ":")),
                "gate_abstained": not diversity or not calibration,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        )
    predictions = build_prediction_rows(
        source_rows,
        diversity_pairs=diversity_pairs,
        calibration_pairs=calibration_pairs,
        expected_datasets=EXPECTED_DATASETS,
    )
    hard_count = sum(len(row["automatic_hard_indices"]) for row in predictions)
    standard_count = sum(len(row["standard_review_indices"]) for row in predictions)
    if hard_count != EXPECTED_HARD_ALERTS or standard_count != EXPECTED_STANDARD_CANDIDATES:
        raise ValueError(f"candidate parity mismatch: Hard={hard_count}, Standard={standard_count}")
    _write_csv(root / "exp303_gate_choices.csv", gate_rows)
    return freeze_rows(
        predictions,
        root / "exp303_frozen_predictions.jsonl",
        root / "exp303_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp293_audit_sha256": EXP293_AUDIT_SHA256,
            "exp294_prediction_sha256": EXP294_PREDICTION_SHA256,
            "train_source_manifests": source_manifests,
            "expansion_policies": list(ALL_POLICIES[1:]),
            "precision_floor_used": False,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
            "retrospective_counterfactual": True,
        },
    )


def _report(results: list[dict[str, object]]) -> str:
    lines = [
        "# Exp303 Expanded TRAIN-only Gates",
        "",
        "Direct verified-expert consensus and two additional TRAIN-only Gates are evaluated without an 80% pass threshold.",
        "No variant is selected from TEST metrics and the operational automatic policy remains unchanged.",
        "",
        "| Policy | Alerts | TP | FP | Precision | Recall | Mean F1 | Added TP | Added FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["scope"] != "all_1117":
            continue
        lines.append(
            f"| {row['expansion_policy']} | {row['automatic_count']} | {row['automatic_tp']} | {row['automatic_fp']} | "
            f"{100 * float(row['automatic_precision']):.3f}% | {100 * float(row['automatic_recall']):.3f}% | "
            f"{float(row['mean_automatic_f1']):.6f} | {row['promoted_tp']} | {row['promoted_fp']} |"
        )
    lines.extend([
        "",
        "- New Gates reuse the same four verified sources; they are not new independent evidence.",
        "- Prediction is frozen before TEST labels are loaded.",
        "- Retrospective only; no operational promotion is activated.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp303_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp303_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp303 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in predictions
    }
    results = evaluate_prediction_rows(predictions, labels_by_dataset=labels)
    _write_csv(root / "exp303_results.csv", results)
    _write_csv(root / "exp303_summary.csv", [row for row in results if row["scope"] == "all_1117"])
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp303_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "prediction_sha256": sha256_file(prediction_path),
                "labels_loaded_after_prediction_freeze": True,
                "precision_floor_used": False,
                "primary_policy_selected": False,
                "automatic_policy_changed": False,
                "retrospective_counterfactual": True,
                "prospective_validated": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return results


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None,
        errors=errors or [], timestamps=[], stdout_log=log, started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Retrospective expanded-Gate frontier; operational Hard unchanged",
        "review_metric_scope": "Frozen Standard candidate counterfactual promotions",
        "retrospective": True,
        "prospective": False,
    })
    publish_status(payload, output_root=root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()
    _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"):
            run_prediction(args.output_root)
        if args.phase in ("evaluate", "all"):
            run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"])
        raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()
