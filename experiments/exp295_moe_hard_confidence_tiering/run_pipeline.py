from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp292_standard_review_evidence_tiering.sources import load_all_source_support
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows
from experiments.exp293_294_train_only_moe_lite.routing import GATE_VARIANTS

from .config import (
    EXPECTED_DATASETS,
    EXPECTED_HARD_ALERTS,
    EXP291_PATH,
    EXP291_SHA256,
    EXP293_ROOT,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
from .pipeline import TIERS, build_prediction_rows, evaluate_prediction_rows


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


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
        for row in rows:
            writer.writerow(row)


def _baseline_rows() -> list[dict[str, object]]:
    observed = sha256_file(EXP291_PATH)
    if observed != EXP291_SHA256:
        raise ValueError(f"Exp291 SHA256 mismatch: {observed}")
    rows = _load_jsonl(EXP291_PATH)
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError(f"Exp291 coverage mismatch: {len(rows)}/{EXPECTED_DATASETS}")
    return rows


def run_prediction(root: Path) -> dict[str, object]:
    baseline = _baseline_rows()
    gate_rows = verify_rows(
        EXP293_ROOT / "exp293_frozen_gate_choices.jsonl",
        EXP293_ROOT / "exp293_gate_manifest.json",
    )
    all_support, source_manifests = load_all_source_support()
    predictions = build_prediction_rows(
        baseline_rows=baseline,
        gate_rows=gate_rows,
        all_support=all_support,
        expected_datasets=EXPECTED_DATASETS,
    )
    if len(predictions) != EXPECTED_DATASETS * len(GATE_VARIANTS):
        raise ValueError("Exp295 prediction coverage mismatch")
    for variant in GATE_VARIANTS:
        rows = [row for row in predictions if row["gate_variant"] == variant]
        hard_count = sum(len(row["automatic_hard_indices"]) for row in rows)
        record_count = sum(len(row["hard_confidence_records"]) for row in rows)
        if hard_count != EXPECTED_HARD_ALERTS or record_count != hard_count:
            raise ValueError(f"Exp295 Hard parity mismatch: {variant}={record_count}/{hard_count}")
    return freeze_rows(
        predictions,
        root / "exp295_frozen_predictions.jsonl",
        root / "exp295_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp291_sha256": EXP291_SHA256,
            "exp293_gate_sha256": sha256_file(EXP293_ROOT / "exp293_frozen_gate_choices.jsonl"),
            "source_manifests": source_manifests,
            "gate_variants": list(GATE_VARIANTS),
            "confidence_tiers": list(TIERS),
            "automatic_policy_changed": False,
            "review_candidate_promoted": False,
            "primary_gate_selected": False,
        },
    )


def _result_report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Exp295 MoE Hard Alert Confidence Tiering",
        "",
        "Exp291 automatic Hard alerts remain unchanged. Confidence tiers are retrospective annotations only.",
        "No Gate variant is selected from TEST metrics.",
        "",
        "| Gate | Tier | Alerts | TP | FP | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["scope"] != "all_1117":
            continue
        lines.append(
            f"| {row['gate_variant']} | {row['confidence_tier']} | {row['hard_count']} | "
            f"{row['hard_tp']} | {row['hard_fp']} | {100 * float(row['hard_precision']):.3f}% | "
            f"{100 * float(row['hard_recall']):.3f}% |"
        )
    lines.extend([
        "",
        "- `all_hard` is autonomous Exp291 performance.",
        "- `high`, `medium`, and `needs_confirmation` do not change the alert lane.",
        "- This is retrospective and not prospectively validated.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp295_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp295_prediction_manifest.json")
    datasets = sorted({str(row["dataset"]) for row in predictions})
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError("Exp295 frozen prediction dataset coverage mismatch")

    labels = {
        dataset: np.asarray(load_test_labels_for_evaluation(dataset), dtype=np.int8)
        for dataset in datasets
    }
    results = evaluate_prediction_rows(predictions=predictions, labels_by_dataset=labels)
    _write_csv(root / "exp295_results.csv", results)
    _write_csv(
        root / "exp295_summary.csv",
        [row for row in results if row["scope"] == "all_1117"],
    )
    (root / "RESULT_REPORT.md").write_text(_result_report(results), encoding="utf-8")
    (root / "exp295_evaluation_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "prediction_sha256": sha256_file(prediction_path),
                "labels_loaded_after_prediction_freeze": True,
                "automatic_policy_changed": False,
                "primary_gate_selected": False,
                "retrospective": True,
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
            "autonomous_metric_scope": "Exp291 Hard unchanged; MoE confidence annotation only",
            "review_metric_scope": "No review promotion or demotion",
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
    root = args.output_root
    started = time.time()
    _publish(root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"):
            run_prediction(root)
        if args.phase in ("evaluate", "all"):
            run_evaluation(root)
    except Exception as exc:
        _publish(root, "failed", args.phase, 0, started, [f"{type(exc).__name__}: {exc}"])
        raise
    _publish(root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()
