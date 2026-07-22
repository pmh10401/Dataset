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
    ALPHA,
    EXPECTED_DATASETS,
    EXP239_MANIFEST_PATH,
    EXP239_PREDICTION_PATH,
    EXP239_PREDICTION_SHA256,
    EXP304_MANIFEST_PATH,
    EXP304_PREDICTION_PATH,
    EXP304_PREDICTION_SHA256,
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


def _load_wavelet_support() -> tuple[dict[str, set[int]], dict[str, object]]:
    if sha256_file(EXP239_PREDICTION_PATH) != EXP239_PREDICTION_SHA256:
        raise ValueError("Exp239 prediction SHA256 mismatch")
    manifest = json.loads(EXP239_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = {
        "datasets": 1117,
        "rows": 4468,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "prediction_sha256": EXP239_PREDICTION_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Exp239 manifest contract mismatch: {key}")
    support: dict[str, set[int]] = {}
    for line in EXP239_PREDICTION_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("variant") != "default" or float(row.get("alpha", -1.0)) != ALPHA:
            continue
        dataset = str(row["dataset"])
        indices = row.get("candidate_indices")
        if dataset in support or not isinstance(indices, list):
            raise ValueError(f"invalid duplicate Exp239 row: {dataset}")
        support[dataset] = set(map(int, indices))
    if len(support) != 1117:
        raise ValueError(f"Exp239 alpha coverage mismatch: {len(support)}/1117")
    return support, manifest


def run_prediction(root: Path) -> dict[str, object]:
    if sha256_file(EXP304_PREDICTION_PATH) != EXP304_PREDICTION_SHA256:
        raise ValueError("Exp304 prediction SHA256 mismatch")
    source_rows = verify_rows(EXP304_PREDICTION_PATH, EXP304_MANIFEST_PATH)
    if len(source_rows) != EXPECTED_DATASETS:
        raise ValueError("Exp304 common-support coverage mismatch")
    wavelet_all, wavelet_manifest = _load_wavelet_support()
    datasets = {str(row["dataset"]) for row in source_rows}
    if not datasets <= set(wavelet_all):
        raise ValueError("Exp239 does not cover all Exp304 datasets")
    predictions = build_prediction_rows(
        source_rows,
        wavelet_support_by_dataset={dataset: wavelet_all[dataset] for dataset in datasets},
        expected_datasets=EXPECTED_DATASETS,
    )
    return freeze_rows(
        predictions,
        root / "exp305_frozen_predictions.jsonl",
        root / "exp305_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "alpha": ALPHA,
            "common_support_datasets": EXPECTED_DATASETS,
            "exp304_prediction_sha256": EXP304_PREDICTION_SHA256,
            "exp239_prediction_sha256": EXP239_PREDICTION_SHA256,
            "exp239_prediction_manifest": wavelet_manifest,
            "policies": list(ALL_POLICIES[1:]),
            "wavelet_full_coverage_verified": True,
            "reconstruction_provenance_verified": False,
            "primary_policy_selected": False,
            "automatic_policy_changed": False,
        },
    )


def _report(results: list[dict[str, object]]) -> str:
    lines = [
        "# Exp305 Wavelet-Reconstruction Cross-confirmation",
        "",
        "This retrospective experiment covers the 251 Exp304 reconstruction common-support datasets.",
        "The operational Hard alert set is frozen and no policy is selected from TEST performance.",
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
        "- Wavelet evidence comes from Exp239 `default`, alpha `0.005`, with full 1,117-dataset coverage.",
        "- Reconstruction remains research-only because Exp257 generator provenance is incomplete.",
        "- TEST labels were loaded only after the Exp305 prediction file was frozen.",
        "",
    ])
    return "\n".join(lines)


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp305_frozen_predictions.jsonl"
    predictions = verify_rows(prediction_path, root / "exp305_prediction_manifest.json")
    if len(predictions) != EXPECTED_DATASETS:
        raise ValueError("Exp305 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in predictions
    }
    results = evaluate_prediction_rows(predictions, labels_by_dataset=labels)
    _write_csv(root / "exp305_results.csv", results)
    _write_csv(root / "exp305_summary.csv", results)
    (root / "RESULT_REPORT.md").write_text(_report(results), encoding="utf-8")
    (root / "exp305_evaluation_manifest.json").write_text(
        json.dumps({
            "experiment_id": EXPERIMENT_ID,
            "prediction_sha256": sha256_file(prediction_path),
            "labels_loaded_after_prediction_freeze": True,
            "common_support_datasets": EXPECTED_DATASETS,
            "wavelet_full_coverage_verified": True,
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
        "autonomous_metric_scope": "251-dataset wavelet/reconstruction counterfactual; operational Hard unchanged",
        "review_metric_scope": "Research-only cross-confirmation of Standard candidates",
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
