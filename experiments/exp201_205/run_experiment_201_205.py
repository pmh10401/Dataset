#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.exp201_205.artifacts import (
    DATA_DIR,
    FrozenInputs,
    freeze_predictions,
    load_alternative_scores,
    load_frozen_inputs,
    load_test_labels_for_evaluation,
    load_verified_predictions,
)
from experiments.exp201_205.core import (
    ALPHA,
    SEED,
    SOURCE_NAMES,
    binary_metrics,
    classify_size_stratum,
    cross_confirmed_candidates,
    empirical_tail_candidates,
    parse_indices,
    three_source_agreement,
    wilson_interval,
)
from experiments.exp201_205.independent_reaggregate import independently_reaggregate, write_manifest

DEFAULT_OUTPUT_ROOT = DATA_DIR / "exp201_205_large_zero_f1_recovery"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
EXPECTED_DATASETS = 251

STAGE_META = {
    "201": ("experiment_201_frozen_large_size_baseline", 251),
    "202": ("experiment_202_alternative_3of3_train_risk_rescue", 251),
    "203": ("experiment_203_current_alternative_cross_confirmation", 251),
    "204": ("experiment_204_size_stratified_shadow_router", 251),
    "205": ("experiment_205_independent_large_zero_f1_frontier", 251),
}


def _excluded(row: dict[str, object]) -> set[int]:
    return set(row["hard_indices"]) | set(row["standard_indices"]) | set(row["priority_indices"])


def predict_dataset(stage: str, inputs: FrozenInputs, dataset: str) -> list[dict[str, object]]:
    base = inputs.exp137[dataset]
    n_train = int(base["n_train"])
    n_test = int(base["n_test"])
    stratum = classify_size_stratum(n_train)
    if stratum is None:
        raise ValueError("dataset outside frozen large-size population")
    excluded = _excluded(base)
    exp190_high = set(inputs.exp190[dataset]["three_tier_high_confidence"]) - excluded
    exp190_standard = set(inputs.exp190[dataset]["three_tier_standard"]) - excluded
    exp198 = set(inputs.exp198[dataset]) - excluded

    alternative = set()
    source_sets: dict[str, set[int]] = {name: set() for name in SOURCE_NAMES}
    if stage in {"202", "203", "204"}:
        scores = load_alternative_scores(inputs, dataset)
        source_sets = {
            source: empirical_tail_candidates(
                scores[f"{source}_train"], scores[f"{source}_test"], ALPHA
            ) - excluded
            for source in SOURCE_NAMES
        }
        alternative = three_source_agreement(
            source_sets["spectral"], source_sets["shapelet"], source_sets["interval"]
        ) - excluded
    cross = cross_confirmed_candidates(
        alternative_indices=alternative,
        exp198_indices=exp198,
        excluded_exp137_indices=excluded,
    )

    variants: list[tuple[str, set[int]]]
    if stage == "201":
        variants = [("frozen_exp190_high_standard", exp190_high | exp190_standard)]
    elif stage == "202":
        variants = [
            ("alternative_3of3_no_hard_alert_gate", alternative if not base["hard_indices"] else set()),
            ("alternative_3of3_all_large", alternative),
        ]
    elif stage == "203":
        variants = [("alternative_3of3_plus_frozen_exp198", cross)]
    elif stage == "204":
        routed = cross if stratum == "100-199" else exp190_high | exp190_standard | cross
        variants = [("size_stratified_shadow_router", routed)]
    else:
        raise ValueError(f"prediction is not defined for stage {stage}")

    rows = []
    for variant, candidates in variants:
        if candidates & excluded:
            raise ValueError(f"candidate overlaps Exp137 lane dataset={dataset}")
        if candidates and (min(candidates) < 0 or max(candidates) >= n_test):
            raise ValueError(f"candidate outside TEST range dataset={dataset}")
        rows.append(
            {
                "experiment_id": STAGE_META[stage][0],
                "dataset": dataset,
                "variant": variant,
                "size_stratum": stratum,
                "n_train": n_train,
                "n_test": n_test,
                "alpha": ALPHA,
                "candidate_indices": sorted(candidates),
                "candidate_count": len(candidates),
                "source_candidate_indices": {key: sorted(value) for key, value in source_sets.items()},
                "alternative_3of3_indices": sorted(alternative),
                "frozen_exp198_indices": sorted(exp198),
                "frozen_exp190_high_indices": sorted(exp190_high),
                "frozen_exp190_standard_indices": sorted(exp190_standard),
                "final_lane": "shadow_hard_diagnostic",
                "autonomous_alert": False,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "uses_family_name": 0,
                "uses_dataset_name_for_policy": 0,
                "uses_test_length_for_policy": 0,
                "uses_zero_f1_membership": 0,
                "retrospective_counterfactual": 1,
                "prospective_validated": 0,
            }
        )
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    normalized = list(rows)
    fields: list[str] = []
    for row in normalized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized)


def _evaluate(stage: str, inputs: FrozenInputs, predictions: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
    results = []
    for prediction in predictions:
        dataset = str(prediction["dataset"])
        labels = load_test_labels_for_evaluation(dataset)
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch dataset={dataset}")
        candidates = parse_indices(prediction["candidate_indices"])
        baseline_hard = set(inputs.exp137[dataset]["hard_indices"])
        added = binary_metrics(labels, candidates)
        baseline = binary_metrics(labels, baseline_hard)
        combined = binary_metrics(labels, baseline_hard | candidates)
        results.append(
            {
                "experiment_id": STAGE_META[stage][0],
                "dataset_name": dataset,
                "variant": prediction["variant"],
                "size_stratum": prediction["size_stratum"],
                "n_train": prediction["n_train"],
                "n_test": prediction["n_test"],
                "anomaly_count": int(np.sum(labels)),
                "candidate_indices": " ".join(map(str, sorted(candidates))),
                "candidate_count": added["count"],
                "candidate_tp": added["tp"],
                "candidate_fp": added["fp"],
                "candidate_precision": added["precision"],
                "candidate_recall": added["recall"],
                "candidate_f1": added["f1"],
                "baseline_hard_count": baseline["count"],
                "baseline_hard_tp": baseline["tp"],
                "baseline_hard_fp": baseline["fp"],
                "baseline_hard_precision": baseline["precision"],
                "baseline_hard_recall": baseline["recall"],
                "baseline_hard_f1": baseline["f1"],
                "counterfactual_count": combined["count"],
                "counterfactual_tp": combined["tp"],
                "counterfactual_fp": combined["fp"],
                "counterfactual_precision": combined["precision"],
                "counterfactual_recall": combined["recall"],
                "counterfactual_f1": combined["f1"],
                "zero_f1_before": int(float(baseline["f1"]) == 0.0),
                "zero_f1_after": int(float(combined["f1"]) == 0.0),
                "zero_f1_fixed": int(float(baseline["f1"]) == 0.0 and float(combined["f1"]) > 0.0),
                "zero_f1_new": int(float(baseline["f1"]) > 0.0 and float(combined["f1"]) == 0.0),
                "metric_scope": "retrospective_shadow_hard_diagnostic",
                "autonomous_exp137_unchanged": 1,
                "selected_for_operation": 0,
            }
        )
    return results, _summarize(results)


def _summarize(results: list[dict]) -> list[dict]:
    summaries = []
    variants = sorted({str(row["variant"]) for row in results})
    for variant in variants:
        variant_rows = [row for row in results if row["variant"] == variant]
        for stratum in ("100-199", "200+", "ALL"):
            rows = variant_rows if stratum == "ALL" else [r for r in variant_rows if r["size_stratum"] == stratum]
            if not rows:
                continue
            count = sum(int(r["candidate_count"]) for r in rows)
            tp = sum(int(r["candidate_tp"]) for r in rows)
            fp = sum(int(r["candidate_fp"]) for r in rows)
            base_tp = sum(int(r["baseline_hard_tp"]) for r in rows)
            base_fp = sum(int(r["baseline_hard_fp"]) for r in rows)
            cf_tp = sum(int(r["counterfactual_tp"]) for r in rows)
            cf_fp = sum(int(r["counterfactual_fp"]) for r in rows)
            workloads = np.asarray([int(r["candidate_count"]) for r in rows], dtype=float)
            lower, upper = wilson_interval(tp, count)
            summaries.append(
                {
                    "variant": variant,
                    "size_stratum": stratum,
                    "datasets": len(rows),
                    "active_datasets": sum(int(r["candidate_count"]) > 0 for r in rows),
                    "candidate_count": count,
                    "candidate_tp": tp,
                    "candidate_fp": fp,
                    "candidate_precision": tp / max(1, count),
                    "candidate_precision_wilson_low": lower,
                    "candidate_precision_wilson_high": upper,
                    "candidate_sample_exploratory": int(count < 30),
                    "retrospective_80pct_point_gate": int(count > 0 and tp / count >= 0.80),
                    "baseline_hard_tp": base_tp,
                    "baseline_hard_fp": base_fp,
                    "baseline_hard_precision": base_tp / max(1, base_tp + base_fp),
                    "baseline_mean_hard_f1": float(np.mean([float(r["baseline_hard_f1"]) for r in rows])),
                    "counterfactual_tp": cf_tp,
                    "counterfactual_fp": cf_fp,
                    "counterfactual_precision": cf_tp / max(1, cf_tp + cf_fp),
                    "counterfactual_mean_f1": float(np.mean([float(r["counterfactual_f1"]) for r in rows])),
                    "zero_f1_before": sum(int(r["zero_f1_before"]) for r in rows),
                    "zero_f1_after": sum(int(r["zero_f1_after"]) for r in rows),
                    "zero_f1_fixed": sum(int(r["zero_f1_fixed"]) for r in rows),
                    "zero_f1_new": sum(int(r["zero_f1_new"]) for r in rows),
                    "workload_mean": float(np.mean(workloads)),
                    "workload_median": float(np.median(workloads)),
                    "workload_p90": float(np.quantile(workloads, 0.90)),
                    "workload_p95": float(np.quantile(workloads, 0.95)),
                    "workload_max": int(np.max(workloads)),
                    "autonomous_exp137_unchanged": 1,
                    "retrospective_counterfactual": 1,
                    "selected_for_operation": 0,
                }
            )
    return summaries


def _paths(stage: str, experiment_id: str, output_root: Path) -> tuple[Path, Path, Path, Path]:
    prediction = output_root / f"exp{stage}_frozen_predictions.jsonl"
    manifest = output_root / f"exp{stage}_prediction_manifest.json"
    result_root = DATA_DIR if output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve() else output_root
    detail = result_root / f"{experiment_id}_results.csv"
    summary = result_root / f"{experiment_id}_summary.csv"
    return prediction, manifest, detail, summary


def run_stage(stage: str, dataset_limit: int | None, output_root: Path, workers: int) -> None:
    if stage == "205":
        experiment_id = STAGE_META[stage][0]
        result_root = DATA_DIR if output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve() else output_root
        stage_files = {
            item: result_root / f"{STAGE_META[item][0]}_results.csv" for item in ("201", "202", "203", "204")
        }
        detail = result_root / f"{experiment_id}_results.csv"
        summary = result_root / f"{experiment_id}_summary.csv"
        independently_reaggregate(
            output_root=output_root, stage_files=stage_files, result_path=detail, summary_path=summary
        )
        completion = output_root / "exp205_completion.json"
        completion.write_text(json.dumps({"experiment_id": experiment_id, "status": "completed", "datasets": 251}, indent=2) + "\n")
        write_manifest(output_root, experiment_id, [detail, summary, completion], {})
        print("Exp205 independent reaggregation completed", flush=True)
        return

    inputs = load_frozen_inputs(dataset_limit=dataset_limit)
    experiment_id = STAGE_META[stage][0]
    print(f"Exp{stage} prediction datasets={len(inputs.datasets)} workers={workers}", flush=True)
    predictions: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(predict_dataset, stage, inputs, name): name for name in inputs.datasets}
        for done, future in enumerate(as_completed(futures), 1):
            dataset = futures[future]
            try:
                predictions.extend(future.result())
            except Exception as exc:
                errors.append({"dataset": dataset, "error": repr(exc)})
            if done % 10 == 0 or done == len(futures):
                print(
                    f"Progress: [{done:4d}/{len(futures)}] "
                    f"rows={len(predictions)} dataset={dataset} errors={len(errors)}",
                    flush=True,
                )
    if errors:
        _write_csv(output_root / f"exp{stage}_errors.csv", errors)
        raise SystemExit(2)
    predictions.sort(key=lambda row: (str(row["dataset"]), str(row["variant"])))
    prediction_path, manifest_path = freeze_predictions(
        stage, experiment_id, predictions, output_root, inputs.input_hashes
    )
    frozen = load_verified_predictions(prediction_path, manifest_path)
    results, summaries = _evaluate(stage, inputs, frozen)
    _, _, detail, summary = _paths(stage, experiment_id, output_root)
    _write_csv(detail, results)
    _write_csv(summary, summaries)
    completion = output_root / f"exp{stage}_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": experiment_id, "status": "completed", "datasets": len(inputs.datasets), "errors": 0}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(
        output_root, experiment_id,
        [prediction_path, manifest_path, detail, summary, completion],
        inputs.input_hashes,
    )
    print(f"Exp{stage} completed datasets={len(inputs.datasets)} rows={len(results)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=list(STAGE_META))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    run_stage(args.stage, args.dataset_limit, args.output_root, args.workers)


if __name__ == "__main__":
    main()
