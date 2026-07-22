from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.artifacts import load_primary_train_scores
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows
from experiments.exp296_cross_gate_hard_confidence_consensus.routing import CONSENSUS_TIERS

from .config import (
    ALPHA,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_DATASETS,
    EXPECTED_HARD_ALERTS,
    EXP295_PREDICTION_SHA256,
    EXP295_ROOT,
    EXP296_PREDICTION_SHA256,
    EXP296_ROOT,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
from .pipeline import build_prediction_rows, evaluate_prediction_rows


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
            writer.writerow({
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _verified_rows(root: Path, filename: str, manifest: str, expected_sha256: str) -> list[dict[str, object]]:
    path = root / filename
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"upstream prediction SHA256 mismatch: {path}={observed}")
    return verify_rows(path, root / manifest)


def _publish(
    root: Path,
    status: str,
    phase: str,
    done: int,
    started: float,
    *,
    current_dataset: str | None = None,
    timestamps: list[tuple[float, int]] | None = None,
    errors: list[str] | None = None,
) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID,
        status,
        phase,
        done,
        EXPECTED_DATASETS,
        current_dataset,
        errors=errors or [],
        timestamps=timestamps or [],
        stdout_log=log,
        started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp291 Hard unchanged; TRAIN-bootstrap confidence stability only",
        "review_metric_scope": "No review promotion or Hard suppression",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "retrospective": True,
        "prospective": False,
    })
    publish_status(payload, output_root=root)


def _alert_rows(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        for record in prediction["stability_records"]:
            rows.append({
                "dataset": prediction["dataset"],
                "candidate_index": record["candidate_index"],
                "consensus_tier": record["consensus_tier"],
                "bootstrap_eligible": record["bootstrap_eligible"],
                "stability_rate": record["stability_rate"],
                "tier_match_count": record["tier_match_count"],
                "bootstrap_replicates": record["bootstrap_replicates"],
                "bootstrap_tier_counts": record["bootstrap_tier_counts"],
                "supporting_gate_count": record["supporting_gate_count"],
                "high_gate_count": record["high_gate_count"],
                "automatic_hard_unchanged": True,
                "contains_test_labels": False,
            })
    return rows


def run_prediction(root: Path, started: float) -> dict[str, object]:
    exp295_rows = _verified_rows(
        EXP295_ROOT,
        "exp295_frozen_predictions.jsonl",
        "exp295_prediction_manifest.json",
        EXP295_PREDICTION_SHA256,
    )
    exp296_rows = _verified_rows(
        EXP296_ROOT,
        "exp296_frozen_predictions.jsonl",
        "exp296_prediction_manifest.json",
        EXP296_PREDICTION_SHA256,
    )
    primary_scores, source_manifests = load_primary_train_scores()
    timestamps: list[tuple[float, int]] = [(time.time(), 0)]

    def progress(dataset: str, done: int) -> None:
        if done == 1 or done % 5 == 0 or done == EXPECTED_DATASETS:
            timestamps.append((time.time(), done))
            if len(timestamps) > 20:
                del timestamps[:-20]
            _publish(
                root,
                "running",
                "bootstrap-predict",
                done,
                started,
                current_dataset=dataset,
                timestamps=timestamps,
            )
        if done % 25 == 0 or done == EXPECTED_DATASETS:
            print(f"[Exp297] {done}/{EXPECTED_DATASETS} dataset={dataset}", flush=True)

    predictions = build_prediction_rows(
        exp296_rows=exp296_rows,
        exp295_rows=exp295_rows,
        primary_train_scores=primary_scores,
        expected_datasets=EXPECTED_DATASETS,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        alpha=ALPHA,
        progress_callback=progress,
    )
    hard_count = sum(len(row["automatic_hard_indices"]) for row in predictions)
    record_count = sum(len(row["stability_records"]) for row in predictions)
    if hard_count != EXPECTED_HARD_ALERTS or record_count != hard_count:
        raise ValueError(f"Exp297 Hard parity mismatch: {record_count}/{hard_count}")
    manifest = freeze_rows(
        predictions,
        root / "exp297_frozen_predictions.jsonl",
        root / "exp297_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp295_prediction_sha256": EXP295_PREDICTION_SHA256,
            "exp296_prediction_sha256": EXP296_PREDICTION_SHA256,
            "source_manifests": source_manifests,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA,
            "automatic_policy_changed": False,
            "operational_policy_selected": False,
        },
    )
    _write_csv(root / "exp297_alert_confidence.csv", _alert_rows(predictions))
    return manifest


def _report(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Exp297 TRAIN-Bootstrap Confidence Stability",
        "",
        f"Bootstrap configuration: {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}.",
        "Automatic Hard alerts remain unchanged. Stability is explanatory only.",
        "",
        "| Tier | Alerts | TP | FP | Precision | Eligible | Abstained | Median stability |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["scope"] != "all_1117":
            continue
        stability = "-" if row["median_stability_rate"] is None else f"{100 * float(row['median_stability_rate']):.1f}%"
        lines.append(
            f"| {row['consensus_tier']} | {row['hard_count']} | {row['hard_tp']} | {row['hard_fp']} | "
            f"{100 * float(row['hard_precision']):.3f}% | {row['eligible_alerts']} | {row['abstained_alerts']} | {stability} |"
        )
    lines.extend([
        "",
        "- TEST labels were loaded only after prediction freeze and SHA256 verification.",
        "- No alert was promoted, demoted, or suppressed.",
        "- Retrospective only; not prospectively validated.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp297_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp297_prediction_manifest.json")
    datasets = sorted({str(row["dataset"]) for row in predictions})
    if len(datasets) != EXPECTED_DATASETS:
        raise ValueError("Exp297 frozen prediction coverage mismatch")
    labels = {
        dataset: np.asarray(load_test_labels_for_evaluation(dataset), dtype=np.int8)
        for dataset in datasets
    }
    results = evaluate_prediction_rows(predictions=predictions, labels_by_dataset=labels)
    _write_csv(root / "exp297_results.csv", results)
    summary = [row for row in results if row["scope"] == "all_1117"]
    _write_csv(root / "exp297_summary.csv", summary)
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp297_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "automatic_policy_changed": False,
        "operational_policy_selected": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    started = time.time()
    _publish(args.output_root, "running", args.phase, 0, started)
    try:
        if args.phase in ("predict", "all"):
            run_prediction(args.output_root, started)
        if args.phase in ("evaluate", "all"):
            _publish(args.output_root, "running", "evaluate", EXPECTED_DATASETS, started)
            run_evaluation(args.output_root)
    except Exception as exc:
        _publish(args.output_root, "failed", args.phase, 0, started, errors=[f"{type(exc).__name__}: {exc}"])
        raise
    _publish(args.output_root, "completed", args.phase, EXPECTED_DATASETS, started)


if __name__ == "__main__":
    main()
