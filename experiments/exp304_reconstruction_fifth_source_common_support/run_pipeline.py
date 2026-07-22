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

from .config import (
    ALPHA,
    EXPECTED_DATASETS,
    EXP257_AUDIT_PATH,
    EXP257_AUDIT_SHA256,
    EXP257_PREDICTION_PATH,
    EXP257_PREDICTION_SHA256,
    EXP294_MANIFEST_PATH,
    EXP294_PREDICTION_PATH,
    EXP294_PREDICTION_SHA256,
    EXPERIMENT_ID,
    OUTPUT_ROOT,
)
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


def _audit_exp257() -> dict[str, object]:
    if sha256_file(EXP257_AUDIT_PATH) != EXP257_AUDIT_SHA256:
        raise ValueError("Exp257 audit SHA256 mismatch")
    if sha256_file(EXP257_PREDICTION_PATH) != EXP257_PREDICTION_SHA256:
        raise ValueError("Exp257 prediction SHA256 mismatch")
    audit = json.loads(EXP257_AUDIT_PATH.read_text(encoding="utf-8"))
    expected = {
        "datasets": EXPECTED_DATASETS,
        "population_rule": "train_normal_count >= 100",
        "contains_test_labels": False,
        "deployment_eligible": False,
        "retrospective": True,
    }
    for key, value in expected.items():
        if audit.get(key) != value:
            raise ValueError(f"Exp257 audit contract mismatch: {key}")
    return audit


def run_prediction(root: Path) -> dict[str, object]:
    reconstruction_audit = _audit_exp257()
    if sha256_file(EXP294_PREDICTION_PATH) != EXP294_PREDICTION_SHA256:
        raise ValueError("Exp294 prediction SHA256 mismatch")
    source_rows = verify_rows(EXP294_PREDICTION_PATH, EXP294_MANIFEST_PATH)
    all_support, source_manifests = load_all_source_support()
    reconstruction = all_support.get("masked_reconstruction", {})
    if len(reconstruction) != EXPECTED_DATASETS:
        raise ValueError(f"reconstruction support coverage mismatch: {len(reconstruction)}/{EXPECTED_DATASETS}")
    predictions = build_prediction_rows(
        source_rows,
        reconstruction_support_by_dataset=reconstruction,
        expected_datasets=EXPECTED_DATASETS,
    )
    hard_count = sum(len(row["automatic_hard_indices"]) for row in predictions)
    standard_count = sum(len(row["standard_review_indices"]) for row in predictions)
    return freeze_rows(
        predictions,
        root / "exp304_frozen_predictions.jsonl",
        root / "exp304_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "alpha": ALPHA,
            "common_support_datasets": EXPECTED_DATASETS,
            "baseline_hard_count": hard_count,
            "standard_candidate_count": standard_count,
            "exp257_audit_sha256": EXP257_AUDIT_SHA256,
            "exp257_prediction_sha256": EXP257_PREDICTION_SHA256,
            "exp294_prediction_sha256": EXP294_PREDICTION_SHA256,
            "source_manifests": source_manifests,
            "exp257_audit": reconstruction_audit,
            "policies": list(ALL_POLICIES[1:]),
            "reconstruction_provenance_verified": False,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
        },
    )


def _report(results: list[dict[str, object]]) -> str:
    lines = [
        "# Exp304 Reconstruction Fifth Source Common-support",
        "",
        "This report covers only the 251 datasets with frozen Exp257 reconstruction support.",
        "Masked reconstruction remains research-only because its dedicated generator provenance is incomplete.",
        "",
        "| Policy | Alerts | TP | FP | Precision | Recall | Mean F1 | Added TP | Added FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['expansion_policy']} | {row['automatic_count']} | {row['automatic_tp']} | {row['automatic_fp']} | "
            f"{100 * float(row['automatic_precision']):.3f}% | {100 * float(row['automatic_recall']):.3f}% | "
            f"{float(row['mean_automatic_f1']):.6f} | {row['promoted_tp']} | {row['promoted_fp']} |"
        )
    lines.extend([
        "",
        "- No missing reconstruction dataset is treated as empty support.",
        "- Alpha and mask variants were not selected from TEST metrics.",
        "- Operational automatic alerts remain unchanged.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp304_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp304_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp304 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in predictions
    }
    results = evaluate_prediction_rows(predictions, labels_by_dataset=labels)
    _write_csv(root / "exp304_results.csv", results)
    _write_csv(root / "exp304_summary.csv", results)
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp304_evaluation_manifest.json").write_text(
        json.dumps({
            "experiment_id": EXPERIMENT_ID,
            "prediction_sha256": sha256_file(prediction_path),
            "labels_loaded_after_prediction_freeze": True,
            "common_support_datasets": EXPECTED_DATASETS,
            "reconstruction_provenance_verified": False,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
            "retrospective_counterfactual": True,
            "prospective_validated": False,
        }, indent=2, sort_keys=True) + "\n",
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
        "autonomous_metric_scope": "251-dataset reconstruction common-support counterfactual; operational Hard unchanged",
        "review_metric_scope": "Research-only fifth-source Standard candidate evidence",
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
