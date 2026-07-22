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
    EXP295_PREDICTION_SHA256,
    EXP295_ROOT,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
from .pipeline import build_prediction_rows, evaluate_prediction_rows
from .routing import CONSENSUS_TIERS


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
    prediction_path = EXP295_ROOT / "exp295_frozen_predictions.jsonl"
    observed = sha256_file(prediction_path)
    if observed != EXP295_PREDICTION_SHA256:
        raise ValueError(f"Exp295 prediction SHA256 mismatch: {observed}")
    exp295_rows = verify_rows(
        prediction_path,
        EXP295_ROOT / "exp295_prediction_manifest.json",
    )
    predictions = build_prediction_rows(
        exp295_rows=exp295_rows,
        expected_datasets=EXPECTED_DATASETS,
    )
    hard_count = sum(len(row["automatic_hard_indices"]) for row in predictions)
    record_count = sum(len(row["consensus_records"]) for row in predictions)
    if hard_count != EXPECTED_HARD_ALERTS or record_count != hard_count:
        raise ValueError(f"Exp296 Hard parity mismatch: {record_count}/{hard_count}")
    return freeze_rows(
        predictions,
        root / "exp296_frozen_predictions.jsonl",
        root / "exp296_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp295_prediction_sha256": observed,
            "consensus_tiers": list(CONSENSUS_TIERS),
            "automatic_policy_changed": False,
            "review_candidate_promoted": False,
            "operational_policy_selected": False,
            "tier_definition_derived_after_exp295_retrospective_analysis": True,
        },
    )


def _report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Exp296 Cross-Gate Hard Confidence Consensus",
        "",
        "Exp291 automatic Hard alerts remain unchanged. The tiers combine all three frozen Exp295 Gate annotations.",
        "This tier design follows retrospective Exp295 analysis and is not an operational suppression policy.",
        "",
        "| Tier | Alerts | TP | FP | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["scope"] != "all_1117":
            continue
        lines.append(
            f"| {row['consensus_tier']} | {row['hard_count']} | {row['hard_tp']} | {row['hard_fp']} | "
            f"{100 * float(row['hard_precision']):.3f}% | {100 * float(row['hard_recall']):.3f}% |"
        )
    lines.extend([
        "",
        "- Every tier remains an autonomous Hard alert.",
        "- No review candidate is promoted and no Hard alert is suppressed.",
        "- TEST labels are loaded only after prediction freeze and hash verification.",
        "- Retrospective only; not prospectively validated.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp296_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp296_prediction_manifest.json")
    datasets = sorted({str(row["dataset"]) for row in predictions})
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError("Exp296 frozen prediction dataset coverage mismatch")
    labels = {
        dataset: np.asarray(load_test_labels_for_evaluation(dataset), dtype=np.int8)
        for dataset in datasets
    }
    results = evaluate_prediction_rows(predictions=predictions, labels_by_dataset=labels)
    _write_csv(root / "exp296_results.csv", results)
    _write_csv(
        root / "exp296_summary.csv",
        [row for row in results if row["scope"] == "all_1117"],
    )
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp296_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "prediction_sha256": sha256_file(prediction_path),
                "labels_loaded_after_prediction_freeze": True,
                "automatic_policy_changed": False,
                "operational_policy_selected": False,
                "retrospective": True,
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
    payload.update({
        "autonomous_metric_scope": "Exp291 Hard unchanged; cross-Gate confidence annotation only",
        "review_metric_scope": "No review promotion or Hard suppression",
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
