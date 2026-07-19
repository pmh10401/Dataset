from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.baseline_exp137 import (
    EXP137_RESULTS,
    load_exp137_baseline,
    verify_exp137_exact_replay,
)
from experiments.exp226_235_hard_alert_quality.database import (
    load_test_labels_for_evaluation,
)
from experiments.exp236_detection_coverage_expansion.core import (
    EXPERIMENT_ID,
    build_expanded_lanes,
    freeze_predictions,
    load_verified_predictions,
    sha256_file,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
SOURCE_ROOT = DATA_ROOT / "exp226_235_hard_alert_quality"
SOURCE_PREDICTION = SOURCE_ROOT / "predictions" / "experiment_234.jsonl"
SOURCE_MANIFEST = SOURCE_ROOT / "manifest" / "experiment_234_prediction.json"
EXPECTED_SOURCE_SHA256 = "6a7ebfd2f6a90982ff9d1103e4dcc21ecfb9d07ee3afd67e8c2225ac3fe94d60"
OUTPUT_ROOT = DATA_ROOT / "exp236_detection_coverage_expansion"
REPO_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs" / "exp236_detection_coverage_expansion"


def _load_exp234_maps() -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]], str]:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    actual = sha256_file(SOURCE_PREDICTION)
    if actual != manifest.get("prediction_sha256") or actual != EXPECTED_SOURCE_SHA256:
        raise ValueError("frozen Exp234 prediction SHA-256 mismatch")
    rows = [
        json.loads(line)
        for line in SOURCE_PREDICTION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("Exp234 prediction row count mismatch")
    by_dataset: dict[str, dict[str, frozenset[int]]] = {}
    abstained: set[str] = set()
    for row in rows:
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp234 source violates information boundary")
        dataset = str(row["dataset"])
        variant = str(row["variant"])
        if variant == "calibration_abstention":
            abstained.add(dataset)
            continue
        by_dataset.setdefault(dataset, {})[variant] = frozenset(
            int(value) for value in row.get("research_indices", ())
        )
    coverage = set(by_dataset) | abstained
    if len(coverage) != 1117 or set(by_dataset) & abstained:
        raise ValueError(f"Exp234 explicit coverage mismatch: {len(coverage)}/1117")
    minimum_two: dict[str, frozenset[int]] = {}
    minimum_three: dict[str, frozenset[int]] = {}
    expected_variants = {"minimum_sources_2", "minimum_sources_3", "minimum_sources_4"}
    for dataset in coverage:
        if dataset in abstained:
            minimum_two[dataset] = frozenset()
            minimum_three[dataset] = frozenset()
            continue
        variants = by_dataset[dataset]
        if set(variants) != expected_variants:
            raise ValueError(f"Exp234 variant coverage mismatch: {dataset}")
        if not variants["minimum_sources_4"] <= variants["minimum_sources_3"] <= variants["minimum_sources_2"]:
            raise ValueError(f"Exp234 source nesting mismatch: {dataset}")
        minimum_two[dataset] = variants["minimum_sources_2"]
        minimum_three[dataset] = variants["minimum_sources_3"]
    return minimum_two, minimum_three, actual


def predict(output_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    baseline = load_exp137_baseline()
    report = verify_exp137_exact_replay(baseline)
    if report.datasets != 1117 or report.lane_overlap_datasets or report.index_error_datasets:
        raise ValueError("Exp137 baseline replay contract failed")
    minimum_two, minimum_three, source_hash = _load_exp234_maps()
    if set(baseline) != set(minimum_two):
        raise ValueError("Exp137/Exp234 dataset coverage mismatch")
    rows: list[dict[str, object]] = []
    for dataset in sorted(baseline):
        lanes = build_expanded_lanes(
            baseline[dataset],
            minimum_two=minimum_two[dataset],
            minimum_three=minimum_three[dataset],
        )
        rows.append({
            "experiment_id": EXPERIMENT_ID,
            "dataset": dataset,
            "n_test": baseline[dataset].test_size,
            "hard_indices": sorted(lanes.hard),
            "standard_review_indices": sorted(lanes.standard_review),
            "priority_review_indices": sorted(lanes.priority_review),
            "new_hard_indices": sorted(lanes.new_hard),
            "new_standard_review_indices": sorted(lanes.new_standard_review),
            "hard_policy": "exp137_union_minimum_three_sources",
            "standard_review_policy": "exp137_union_exactly_two_sources",
            "priority_review_policy": "unchanged_exp137_priority_review",
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "retrospective_candidate_packaging": True,
            "deployment_eligible": False,
        })
    return freeze_predictions(
        rows,
        output_root,
        input_hashes={
            "exp137_results": sha256_file(EXP137_RESULTS),
            "exp234_predictions": source_hash,
            "exp234_manifest": sha256_file(SOURCE_MANIFEST),
        },
    )


def _metrics(indices: list[int], labels: np.ndarray) -> dict[str, float | int]:
    selected = set(int(value) for value in indices)
    if selected and (min(selected) < 0 or max(selected) >= len(labels)):
        raise ValueError("Exp236 candidate index outside TEST range")
    truth = set(np.flatnonzero(labels == 1).astype(int))
    tp = len(selected & truth); fp = len(selected - truth); fn = len(truth - selected)
    denominator = 2 * tp + fp + fn
    return {
        "count": len(selected), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / denominator if denominator else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(output_root: Path = OUTPUT_ROOT) -> tuple[Path, Path]:
    prediction = Path(output_root) / "exp236_frozen_predictions.jsonl"
    manifest = Path(output_root) / "exp236_prediction_manifest.json"
    rows = load_verified_predictions(prediction, manifest)
    evaluated: list[dict[str, object]] = []
    for row in rows:
        labels = np.asarray(load_test_labels_for_evaluation(str(row["dataset"])), dtype=np.int8)
        if len(labels) != int(row["n_test"]):
            raise ValueError(f"Exp236 TEST length mismatch: {row['dataset']}")
        hard = _metrics(row["hard_indices"], labels)
        standard = _metrics(row["standard_review_indices"], labels)
        priority = _metrics(row["priority_review_indices"], labels)
        new_hard = _metrics(row["new_hard_indices"], labels)
        new_standard = _metrics(row["new_standard_review_indices"], labels)
        evaluated.append({
            **row,
            "true_anomalies": int(np.sum(labels == 1)),
            **{f"hard_{key}": value for key, value in hard.items()},
            **{f"standard_review_{key}": value for key, value in standard.items()},
            **{f"priority_review_{key}": value for key, value in priority.items()},
            **{f"new_hard_{key}": value for key, value in new_hard.items()},
            **{f"new_standard_review_{key}": value for key, value in new_standard.items()},
            "hard_metric_scope": "retrospective_autonomous_counterfactual",
            "review_metric_scope": "human_assisted_diagnostic_only",
        })
    hard_tp = sum(int(row["hard_tp"]) for row in evaluated)
    hard_fp = sum(int(row["hard_fp"]) for row in evaluated)
    hard_fn = sum(int(row["hard_fn"]) for row in evaluated)
    standard_tp = sum(int(row["standard_review_tp"]) for row in evaluated)
    standard_fp = sum(int(row["standard_review_fp"]) for row in evaluated)
    priority_tp = sum(int(row["priority_review_tp"]) for row in evaluated)
    priority_fp = sum(int(row["priority_review_fp"]) for row in evaluated)
    summary = [{
        "experiment_id": EXPERIMENT_ID,
        "variant": "three_source_hard_two_source_review",
        "datasets": len(evaluated),
        "hard_alerts": hard_tp + hard_fp,
        "hard_tp": hard_tp,
        "hard_fp": hard_fp,
        "hard_fn": hard_fn,
        "hard_precision": hard_tp / max(1, hard_tp + hard_fp),
        "hard_recall": hard_tp / max(1, hard_tp + hard_fn),
        "mean_hard_f1": float(np.mean([float(row["hard_f1"]) for row in evaluated])),
        "standard_review_candidates": standard_tp + standard_fp,
        "standard_review_tp": standard_tp,
        "standard_review_fp": standard_fp,
        "standard_review_precision": standard_tp / max(1, standard_tp + standard_fp),
        "priority_review_candidates": priority_tp + priority_fp,
        "priority_review_tp": priority_tp,
        "priority_review_fp": priority_fp,
        "priority_review_precision": priority_tp / max(1, priority_tp + priority_fp),
        "new_hard_tp": sum(int(row["new_hard_tp"]) for row in evaluated),
        "new_hard_fp": sum(int(row["new_hard_fp"]) for row in evaluated),
        "new_standard_review_tp": sum(int(row["new_standard_review_tp"]) for row in evaluated),
        "new_standard_review_fp": sum(int(row["new_standard_review_fp"]) for row in evaluated),
        "policy_selected_from_exp236_test_metrics": 0,
        "selected_from_prior_test_review": 1,
        "retrospective": 1,
        "deployment_eligible": 0,
    }]
    result_path = Path(output_root) / "experiment_236_results.csv"
    summary_path = Path(output_root) / "experiment_236_summary.csv"
    _write_csv(result_path, evaluated); _write_csv(summary_path, summary)
    return result_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.stage in {"predict", "all"}:
        prediction, manifest = predict(args.output_root)
        print(prediction); print(manifest)
    if args.stage in {"evaluate", "all"}:
        result, summary = evaluate(args.output_root)
        print(result); print(summary)


if __name__ == "__main__":
    main()
