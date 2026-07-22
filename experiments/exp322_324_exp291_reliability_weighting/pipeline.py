from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.exp309_316_fn_recovery.contracts import verify_prediction_rows
from experiments.exp309_316_fn_recovery.reporting import write_csv
from experiments.exp317_318_train_reliability_fusion.contracts import (
    freeze_jsonl, sha256_file, verify_frozen_jsonl,
)

from .config import EXP116_RESULTS, EXP291_ROOT, EXPERIMENT_IDS, OUTPUT_ROOTS


def _parse_indices(value) -> list[int]:
    text = str(value or "").strip()
    return sorted({int(float(item)) for item in text.split()}) if text else []


def compare_selected_indices(baseline, adaptive) -> dict[str, object]:
    left = set(_parse_indices(baseline)); right = set(_parse_indices(adaptive))
    return {
        "baseline_indices": sorted(left), "adaptive_indices": sorted(right),
        "changed": left != right, "removed_indices": sorted(left - right),
        "added_indices": sorted(right - left),
    }


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _status(root: Path, experiment: int, status: str, phase: str, done: int, total: int, started: float) -> None:
    _atomic_json(root / "status.json", {
        "id": EXPERIMENT_IDS[experiment], "experiment_id": EXPERIMENT_IDS[experiment],
        "status": status, "phase": phase, "datasets_done": done, "datasets_total": total,
        "progress_percent": 100.0 * done / total if total else 0.0,
        "elapsed_seconds": time.time() - started, "updated_at_unix": time.time(),
        "automatic_policy_changed": False, "retrospective": True,
    })


def run_exp322(root: Path) -> list[dict[str, object]]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    _status(root, 322, "running", "audit", 0, 6, started)
    rows = [
        {"stage": "Exp93 primary rank fusion", "evidence": "temporal random kernel; frequency image; texture shape", "mechanism": "fixed numeric rank weights 0.36/0.22/0.22", "train_reliability_weighted": False, "limitation": "weights are fixed"},
        {"stage": "Exp93 auxiliary representation", "evidence": "Exp84 family-guard and count-cap orders", "mechanism": "fixed weights 0.12/0.08 plus TRAIN-exceed safety gate", "train_reliability_weighted": False, "limitation": "full TRAIN score vectors not retained in Exp87 artifact"},
        {"stage": "Exp116 reliability replay", "evidence": "three primary Exp93 sources", "mechanism": "TRAIN tail, bootstrap and exceed reliability adaptive weights", "train_reliability_weighted": True, "limitation": "not adopted by final chain; only three of five sources"},
        {"stage": "Exp236 expansion", "evidence": "three-source independent evidence", "mechanism": "set consensus, minimum three sources", "train_reliability_weighted": False, "limitation": "gate rather than numeric weighted score"},
        {"stage": "Exp253 expansion", "evidence": "MiniROCKET guarded agreement", "mechanism": "intersection and TRAIN-safe guard", "train_reliability_weighted": False, "limitation": "gate rather than numeric weighted score"},
        {"stage": "Exp291 routing", "evidence": "Exp290 guarded Hard and Exp272 review lanes", "mechanism": "set partition and lane freeze", "train_reliability_weighted": False, "limitation": "no numeric score fusion exists at this final stage"},
    ]
    for row in rows:
        row.update({"experiment_id": EXPERIMENT_IDS[322], "retrospective": True})
    write_csv(root / "exp322_results.csv", rows)
    summary = [{
        "experiment_id": EXPERIMENT_IDS[322], "stages": len(rows),
        "weighted_stages": sum(bool(row["train_reliability_weighted"]) for row in rows),
        "all_upstream_sources_reliability_weighted": False,
        "exp291_final_stage_has_numeric_weights": False,
        "exp116_results_sha256": sha256_file(EXP116_RESULTS), "retrospective": True,
    }]
    write_csv(root / "exp322_summary.csv", summary)
    _atomic_json(root / "exp322_audit_manifest.json", summary[0])
    _status(root, 322, "completed", "audit", 6, 6, started)
    return rows


def _read_exp116() -> dict[str, dict[str, dict[str, str]]]:
    selected: dict[str, dict[str, dict[str, str]]] = {}
    with EXP116_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            config = str(row["config_name"]); dataset = str(row["dataset_name"])
            if config not in {"baseline_exp93", "train_only_reliability_weak_replace"}:
                continue
            selected.setdefault(dataset, {})[config] = row
    return selected


def run_exp323(root: Path) -> dict[str, object]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    source = _read_exp116(); _status(root, 323, "running", "replay", 0, len(source), started)
    if len(source) != 1117 or any(len(rows) != 2 for rows in source.values()):
        raise ValueError("Exp116 baseline/adaptive coverage mismatch")
    outputs = []
    for position, dataset in enumerate(sorted(source), 1):
        baseline = source[dataset]["baseline_exp93"]
        adaptive = source[dataset]["train_only_reliability_weak_replace"]
        comparison = compare_selected_indices(baseline["selected_indices"], adaptive["selected_indices"])
        outputs.append({
            "experiment_id": EXPERIMENT_IDS[323], "dataset": dataset, **comparison,
            "rocket_weight": float(adaptive["rocket_weight"]),
            "frequency_image_weight": float(adaptive["exp55_weight"]),
            "texture_shape_weight": float(adaptive["exp56_weight"]),
            "rocket_reliability": float(adaptive["rocket_reliability"]),
            "frequency_image_reliability": float(adaptive["exp55_reliability"]),
            "texture_shape_reliability": float(adaptive["exp56_reliability"]),
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "uses_test_metrics_for_weights": False, "partial_three_source_scope": True,
            "retrospective": True,
        })
        if position % 25 == 0 or position == len(source):
            _status(root, 323, "running", "replay", position, len(source), started)
    manifest = freeze_jsonl(
        outputs, root / "exp323_frozen_predictions.jsonl", root / "exp323_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[323], metadata={
            "datasets": len(outputs), "changed_datasets": sum(bool(row["changed"]) for row in outputs),
            "exp116_results_sha256": sha256_file(EXP116_RESULTS),
            "partial_three_source_scope": True, "all_five_sources_weighted": False,
            "automatic_policy_changed": False, "retrospective": True,
        },
    )
    write_csv(root / "exp323_results.csv", outputs)
    write_csv(root / "exp323_summary.csv", [{
        "experiment_id": EXPERIMENT_IDS[323], "datasets": len(outputs),
        "changed_datasets": sum(bool(row["changed"]) for row in outputs),
        "unchanged_datasets": sum(not bool(row["changed"]) for row in outputs),
        "all_five_sources_weighted": False, "automatic_policy_changed": False,
    }])
    _status(root, 323, "completed", "replay", len(source), len(source), started)
    return manifest


def run_exp324(root: Path, exp323_root: Path) -> dict[str, object]:
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.time()
    replay = verify_frozen_jsonl(
        Path(exp323_root) / "exp323_frozen_predictions.jsonl",
        Path(exp323_root) / "exp323_prediction_manifest.json",
    )
    changed = [row for row in replay if row["changed"]]
    if changed:
        raise ValueError("Exp116 changed upstream indices; deterministic downstream replay is required")
    baseline = verify_prediction_rows(
        EXP291_ROOT / "exp291_frozen_operational_routing.jsonl",
        EXP291_ROOT / "exp291_prediction_manifest.json",
    )
    _status(root, 324, "running", "route_parity", 0, len(baseline), started)
    outputs = []
    for position, row in enumerate(baseline, 1):
        outputs.append({
            **row, "experiment_id": EXPERIMENT_IDS[324],
            "reliability_weighted_upstream_selection_changed": False,
            "automatic_policy_changed": False,
        })
        if position % 25 == 0 or position == len(baseline):
            _status(root, 324, "running", "route_parity", position, len(baseline), started)
    manifest = freeze_jsonl(
        outputs, root / "exp324_frozen_predictions.jsonl", root / "exp324_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[324], metadata={
            "datasets": len(outputs), "exp323_sha256": sha256_file(Path(exp323_root) / "exp323_frozen_predictions.jsonl"),
            "exp291_sha256": sha256_file(EXP291_ROOT / "exp291_frozen_operational_routing.jsonl"),
            "automatic_hard_exact_match_exp291": True, "all_lanes_exact_match_exp291": True,
            "automatic_policy_changed": False, "retrospective": True,
        },
    )
    write_csv(root / "exp324_results.csv", [{
        "experiment_id": EXPERIMENT_IDS[324], "dataset": row["dataset"],
        "hard_count": len(row["automatic_hard_indices"]),
        "review_count": len(row["default_review_indices"]), "all_lanes_match": True,
    } for row in outputs])
    write_csv(root / "exp324_summary.csv", [{
        "experiment_id": EXPERIMENT_IDS[324], "datasets": len(outputs),
        "automatic_hard_count": sum(len(row["automatic_hard_indices"]) for row in outputs),
        "all_lanes_exact_match_exp291": True, "automatic_policy_changed": False,
        "partial_three_source_scope": True,
    }])
    _status(root, 324, "completed", "route_parity", len(baseline), len(baseline), started)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--experiment", type=int, choices=(322,323,324), required=True)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--exp323-root", type=Path, default=OUTPUT_ROOTS[323])
    args = parser.parse_args(); root = args.output_root or OUTPUT_ROOTS[args.experiment]
    if args.experiment == 322: run_exp322(root)
    elif args.experiment == 323: run_exp323(root)
    else: run_exp324(root, args.exp323_root)


if __name__ == "__main__": main()
