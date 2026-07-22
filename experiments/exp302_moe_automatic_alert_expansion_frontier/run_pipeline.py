from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows

from .config import (
    EXPECTED_DATASETS,
    EXPECTED_HARD_ALERTS,
    EXPECTED_STANDARD_CANDIDATES,
    EXP294_MANIFEST_PATH,
    EXP294_PREDICTION_PATH,
    EXP294_PREDICTION_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
from .pipeline import ALL_POLICIES, build_prediction_rows, evaluate_prediction_rows
from .routing import EXPANSION_POLICIES


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


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(EXP294_PREDICTION_PATH) != EXP294_PREDICTION_SHA256:
        raise ValueError("Exp294 frozen prediction SHA256 mismatch")
    source_rows = verify_rows(EXP294_PREDICTION_PATH, EXP294_MANIFEST_PATH)
    predictions = build_prediction_rows(source_rows, expected_datasets=EXPECTED_DATASETS)
    hard_count = sum(len(row["automatic_hard_indices"]) for row in predictions)
    standard_count = sum(len(row["standard_review_indices"]) for row in predictions)
    if hard_count != EXPECTED_HARD_ALERTS:
        raise ValueError(f"baseline Hard parity mismatch: {hard_count}/{EXPECTED_HARD_ALERTS}")
    if standard_count != EXPECTED_STANDARD_CANDIDATES:
        raise ValueError(f"Standard candidate parity mismatch: {standard_count}/{EXPECTED_STANDARD_CANDIDATES}")
    for row in predictions:
        promoted = row["promoted_indices_by_policy"]
        for narrow, broad in zip(EXPANSION_POLICIES, EXPANSION_POLICIES[1:], strict=False):
            if not set(promoted[narrow]) <= set(promoted[broad]):
                raise ValueError(f"non-nested promotion in {row['dataset']}")
    return freeze_rows(
        predictions,
        root / "exp302_frozen_predictions.jsonl",
        root / "exp302_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp294_prediction_sha256": EXP294_PREDICTION_SHA256,
            "expansion_policies": list(EXPANSION_POLICIES),
            "precision_floor_used": False,
            "global_top_n_used": False,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
            "retrospective_counterfactual": True,
        },
    )


def _report(results: list[dict[str, object]]) -> str:
    lines = [
        "# Exp302 MoE Automatic Alert Expansion Frontier",
        "",
        "Precision 80% is not a pass/fail threshold. No expansion policy is selected from TEST metrics.",
        "All rows are retrospective counterfactuals; the current automatic policy remains unchanged.",
        "",
        "| Expansion policy | Alerts | TP | FP | Precision | Recall | Mean F1 | Added TP | Added FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        if row["scope"] != "all_1117":
            continue
        lines.append(
            f"| {row['expansion_policy']} | {row['automatic_count']} | {row['automatic_tp']} | "
            f"{row['automatic_fp']} | {100 * float(row['automatic_precision']):.3f}% | "
            f"{100 * float(row['automatic_recall']):.3f}% | {float(row['mean_automatic_f1']):.6f} | "
            f"{row['promoted_tp']} | {row['promoted_fp']} |"
        )
    lines.extend(
        [
            "",
            "- No TEST-based 80% gate is applied.",
            "- No global Top-N, TEST-length budget, or family policy is used.",
            "- The result does not update the operational automatic-alert policy.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp302_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp302_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp302 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in predictions
    }
    results = evaluate_prediction_rows(predictions, labels_by_dataset=labels)
    if {row["expansion_policy"] for row in results if row["scope"] == "all_1117"} != set(ALL_POLICIES):
        raise ValueError("Exp302 policy result coverage mismatch")
    _write_csv(root / "exp302_results.csv", results)
    _write_csv(root / "exp302_summary.csv", [row for row in results if row["scope"] == "all_1117"])
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp302_evaluation_manifest.json").write_text(
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
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors: list[str] | None = None) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID,
        status,
        phase,
        done,
        EXPECTED_DATASETS,
        None,
        errors=errors or [],
        timestamps=[],
        stdout_log=log,
        started_unix=started,
    )
    payload.update(
        {
            "autonomous_metric_scope": "Retrospective MoE expansion frontier; operational Hard unchanged",
            "review_metric_scope": "Frozen Standard candidates considered as counterfactual promotions",
            "retrospective": True,
            "prospective": False,
        }
    )
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
