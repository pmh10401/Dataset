from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp226_235_hard_alert_quality.preprocessing import preprocessing_contract
from experiments.exp267_273_zero_support_fn_recovery.reporting import build_status, publish_status
from experiments.exp293_294_train_only_moe_lite.contracts import freeze_rows, sha256_file, verify_rows

from .config import (
    EXPECTED_DATASETS, EXPECTED_STANDARD, EXP307_MANIFEST, EXP307_PREDICTION,
    EXP307_PREDICTION_SHA256, EXPERIMENT_ID, GATE_VARIANT, MEDIUM_COUNT,
    OUTPUT_ROOT, STRONG_COUNT,
)
from .routing import partition_review_records


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


def _load_exp307_rows() -> list[dict[str, object]]:
    if sha256_file(EXP307_PREDICTION) != EXP307_PREDICTION_SHA256:
        raise ValueError("Exp307 frozen prediction SHA256 mismatch")
    rows = verify_rows(EXP307_PREDICTION, EXP307_MANIFEST)
    selected = [row for row in rows if row.get("gate_variant") == GATE_VARIANT]
    if len(selected) != EXPECTED_DATASETS:
        raise ValueError("Exp307 selected Gate coverage mismatch")
    return selected


def run_prediction(root: Path) -> dict[str, object]:
    source_rows = _load_exp307_rows()
    records = [dict(record) for row in source_rows for record in row["candidate_rank_records"]]
    if len(records) != EXPECTED_STANDARD:
        raise ValueError("Exp307 Standard review universe mismatch")
    partitioned = partition_review_records(
        records, strong_count=STRONG_COUNT, medium_count=MEDIUM_COUNT,
    )
    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in partitioned:
        by_dataset[str(record["dataset"])].append(record)
    outputs: list[dict[str, object]] = []
    for source in source_rows:
        dataset = str(source["dataset"])
        dataset_records = sorted(by_dataset[dataset], key=lambda row: int(row["global_review_rank"]))
        tier_indices = {
            tier: [int(row["candidate_index"]) for row in dataset_records if row["review_tier"] == tier]
            for tier in ("strong_review", "medium_review", "standard_review")
        }
        all_indices = set().union(*(set(values) for values in tier_indices.values()))
        if all_indices != set(map(int, source["standard_review_indices"])):
            raise ValueError(f"review candidate parity mismatch: {dataset}")
        outputs.append({
            "experiment_id": EXPERIMENT_ID,
            "dataset": dataset,
            "n_test": int(source["n_test"]),
            "gate_variant": GATE_VARIANT,
            "core_hard_indices": source["core_hard_indices"],
            "benchmark_union_hard_indices": source["benchmark_union_hard_indices"],
            "strong_review_indices": tier_indices["strong_review"],
            "medium_review_indices": tier_indices["medium_review"],
            "standard_review_indices": tier_indices["standard_review"],
            "review_rank_records": dataset_records,
            "automatic_policy_changed": False,
            "review_candidate_set_changed": False,
            "review_tier_policy_frozen": True,
            "deployment_active": False,
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        })
    counts = {
        tier: sum(len(row[f"{tier}_indices"]) for row in outputs)
        for tier in ("strong_review", "medium_review", "standard_review")
    }
    expected = {
        "strong_review": STRONG_COUNT,
        "medium_review": MEDIUM_COUNT,
        "standard_review": EXPECTED_STANDARD - STRONG_COUNT - MEDIUM_COUNT,
    }
    if counts != expected:
        raise ValueError(f"frozen review tier mismatch: {counts} != {expected}")
    _write_csv(root / "exp308_review_candidates.csv", partitioned)
    return freeze_rows(
        outputs,
        root / "exp308_frozen_review_tiers.jsonl",
        root / "exp308_prediction_manifest.json",
        experiment_id=EXPERIMENT_ID,
        metadata={
            "exp307_prediction_sha256": EXP307_PREDICTION_SHA256,
            "gate_variant": GATE_VARIANT,
            "tier_counts": counts,
            "preprocessing_contract": preprocessing_contract(),
            "gate_selected_by_test_metrics": False,
            "review_only": True,
            "automatic_policy_changed": False,
            "deployment_active": False,
            "retrospective": True,
            "prospective_validated": False,
        },
    )


def run_evaluation(root: Path) -> list[dict[str, object]]:
    prediction_path = root / "exp308_frozen_review_tiers.jsonl"
    rows = verify_rows(prediction_path, root / "exp308_prediction_manifest.json")
    if len(rows) != EXPECTED_DATASETS:
        raise ValueError("Exp308 prediction coverage mismatch")
    labels = {
        str(row["dataset"]): np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        for row in rows
    }
    summary: list[dict[str, object]] = []
    for tier in ("strong_review", "medium_review", "standard_review"):
        pairs = [
            (str(row["dataset"]), int(index))
            for row in rows for index in row[f"{tier}_indices"]
        ]
        tp = sum(int(labels[dataset][index] == 1) for dataset, index in pairs)
        summary.append({
            "experiment_id": EXPERIMENT_ID,
            "gate_variant": GATE_VARIANT,
            "review_tier": tier,
            "count": len(pairs),
            "tp": tp,
            "fp": len(pairs) - tp,
            "precision": tp / len(pairs) if pairs else 0.0,
            "autonomous": False,
            "human_assisted_only": True,
            "automatic_policy_changed": False,
            "deployment_active": False,
            "retrospective": True,
        })
    _write_csv(root / "exp308_results.csv", summary)
    _write_csv(root / "exp308_summary.csv", summary)
    report = [
        "# Exp308 Frozen Six-expert Review Tiers", "",
        "Exp307 risk-first ranking is frozen into review-only tiers. Automatic Hard alerts are unchanged.", "",
        "| Tier | Count | TP | FP | Precision |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        report.append(
            f"| {row['review_tier']} | {row['count']} | {row['tp']} | {row['fp']} | {100*float(row['precision']):.3f}% |"
        )
    report.extend([
        "", "- TEST labels were loaded only after the prediction hash was frozen.",
        "- No review tier is promoted to autonomous Hard.",
        "- This is a retrospective candidate policy and is not deployment active.", "",
    ])
    (root / "RESULT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (root / "exp308_evaluation_manifest.json").write_text(json.dumps({
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": sha256_file(prediction_path),
        "labels_loaded_after_prediction_freeze": True,
        "automatic_policy_changed": False,
        "deployment_active": False,
        "retrospective": True,
        "prospective_validated": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _publish(root: Path, status: str, phase: str, done: int, started: float, errors=None) -> None:
    log = root / "logs" / f"{EXPERIMENT_ID}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    payload = build_status(
        EXPERIMENT_ID, status, phase, done, EXPECTED_DATASETS, None,
        errors=errors or [], timestamps=[], stdout_log=log, started_unix=started,
    )
    payload.update({
        "autonomous_metric_scope": "Exp306 core and benchmark Hard unchanged",
        "review_metric_scope": "Exp307 risk-first ranking frozen as review-only tiers",
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
