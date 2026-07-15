#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Add repository root to python path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp206_210.core import (
    ALPHAS,
    SEED,
    SHADOW_RISK,
    REVIEW_RISK,
    SOURCE_NAMES,
    classify_risk,
    empirical_upper_pvalues,
    leave_one_out_pvalues,
    robust_z_pair,
    top2_mean,
    inverse_spearman_weights,
    fisher_stat,
    wilson_upper,
    sha256_file,
    validate_prediction_payload,
    fold_plan,
)

from experiments.exp206_210.alternative_oof import (
    fit_score_source,
    crossfit_alternative_sources,
)

from experiments.exp206_210.artifacts import (
    DATA_DIR,
    DB_PATH,
    EXP206_CACHE_DIR,
    EXP206_MANIFEST,
    EXP206_ROOT,
    load_population_values_only,
    load_dataset_series_only,
    load_current_oof_cache,
    load_exp206_cache,
    load_test_labels_for_evaluation,
    freeze_predictions,
)

from experiments.exp206_210.independent_reaggregate import independently_reaggregate, write_manifest

STAGE_META = {
    "206": ("experiment_206_alternative_source_oof_reconstruction", 251),
    "207": ("experiment_207_train_oof_conditional_novelty_audit", 251),
    "208": ("experiment_208_continuous_alternative_source_fusion", 251),
    "209": ("experiment_209_seven_source_joint_normal_novelty", 251),
    "210": ("experiment_210_independent_frontier_reaggregation", 251),
}


# For 7-source joint normal novelty
def rank_gaussian_diagonal_score(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = train.shape[0]
    from statistics import NormalDist

    train_rg = np.zeros_like(train, dtype=float)
    for j in range(train.shape[1]):
        from experiments.exp206_210.core import rankdata_average
        ranks = rankdata_average(train[:, j])
        quantiles = (ranks - 0.5) / n
        quantiles = np.clip(quantiles, 1e-9, 1.0 - 1e-9)
        train_rg[:, j] = [NormalDist().inv_cdf(q) for q in quantiles]

    query_rg = np.zeros_like(query, dtype=float)
    for j in range(query.shape[1]):
        sorted_col = np.sort(train[:, j])
        ranks = np.searchsorted(sorted_col, query[:, j], side="right")
        quantiles = (ranks - 0.5) / n
        quantiles = np.clip(quantiles, 1e-9, 1.0 - 1e-9)
        query_rg[:, j] = [NormalDist().inv_cdf(q) for q in quantiles]

    medians = np.median(train_rg, axis=0)
    mads = np.median(np.abs(train_rg - medians), axis=0) + 1e-6

    train_scaled = (train_rg - medians) / mads
    query_scaled = (query_rg - medians) / mads

    train_dist = np.sum(train_scaled**2, axis=1)
    query_dist = np.sum(query_scaled**2, axis=1)
    return train_dist, query_dist


def shrinkage_covariance_score(train: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    n = train.shape[0]
    if n < 20:
        return None
    from statistics import NormalDist

    train_rg = np.zeros_like(train, dtype=float)
    for j in range(train.shape[1]):
        from experiments.exp206_210.core import rankdata_average
        ranks = rankdata_average(train[:, j])
        quantiles = (ranks - 0.5) / n
        quantiles = np.clip(quantiles, 1e-9, 1.0 - 1e-9)
        train_rg[:, j] = [NormalDist().inv_cdf(q) for q in quantiles]

    query_rg = np.zeros_like(query, dtype=float)
    for j in range(query.shape[1]):
        sorted_col = np.sort(train[:, j])
        ranks = np.searchsorted(sorted_col, query[:, j], side="right")
        quantiles = (ranks - 0.5) / n
        quantiles = np.clip(quantiles, 1e-9, 1.0 - 1e-9)
        query_rg[:, j] = [NormalDist().inv_cdf(q) for q in quantiles]

    emp_cov = np.cov(train_rg, rowvar=False)
    if not np.all(np.isfinite(emp_cov)):
        return None

    diag_cov = np.diag(np.diag(emp_cov))
    shrunk_cov = 0.8 * emp_cov + 0.2 * diag_cov

    try:
        inv_cov = np.linalg.inv(shrunk_cov)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(inv_cov)):
        return None

    mean_rg = np.mean(train_rg, axis=0)
    train_diff = train_rg - mean_rg
    query_diff = query_rg - mean_rg

    train_dist = np.sum(train_diff @ inv_cov * train_diff, axis=1)
    query_dist = np.sum(query_diff @ inv_cov * query_diff, axis=1)

    return train_dist, query_dist


def compute_fused_train_test(
    alt_scores: dict[str, np.ndarray], method: str
) -> tuple[np.ndarray, np.ndarray]:
    train_mat = np.column_stack([alt_scores[f"{s}_train"] for s in ["spectral", "shapelet", "interval"]])
    test_mat = np.column_stack([alt_scores[f"{s}_test"] for s in ["spectral", "shapelet", "interval"]])

    train_z = np.zeros_like(train_mat)
    test_z = np.zeros_like(test_mat)
    for col in range(3):
        tr_z, te_z = robust_z_pair(train_mat[:, col], test_mat[:, col])
        train_z[:, col] = tr_z
        test_z[:, col] = te_z

    if method == "top2_mean":
        fused_train = top2_mean(train_z)
        fused_test = top2_mean(test_z)
    elif method == "weighted_top2":
        weights = inverse_spearman_weights(train_z)
        fused_train = top2_mean(train_z * weights)
        fused_test = top2_mean(test_z * weights)
    elif method == "fisher":
        p_train = np.column_stack([leave_one_out_pvalues(train_mat[:, col]) for col in range(3)])
        p_test = np.column_stack([empirical_upper_pvalues(train_mat[:, col], test_mat[:, col]) for col in range(3)])
        fused_train = fisher_stat(p_train)
        fused_test = fisher_stat(p_test)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return fused_train, fused_test


def compute_joint_train_test(
    current_scores: dict[str, np.ndarray], alt_scores: dict[str, np.ndarray], method: str
) -> tuple[np.ndarray, np.ndarray] | None:
    srcs = ["rocket_exp40", "exp55_best", "exp56_best", "exp84", "spectral", "shapelet", "interval"]
    train_parts = []
    test_parts = []
    for s in srcs:
        if s in ["rocket_exp40", "exp55_best", "exp56_best", "exp84"]:
            train_parts.append(current_scores[f"{s}_train"])
            test_parts.append(current_scores[f"{s}_test"])
        else:
            train_parts.append(alt_scores[f"{s}_train"])
            test_parts.append(alt_scores[f"{s}_test"])

    train_mat = np.column_stack(train_parts)
    test_mat = np.column_stack(test_parts)

    if method == "rank_gaussian_diagonal":
        return rank_gaussian_diagonal_score(train_mat, test_mat)
    elif method == "shrinkage_covariance":
        return shrinkage_covariance_score(train_mat, test_mat)
    else:
        raise ValueError(f"Unknown joint method: {method}")


def compute_oof_false_alerts(fused_train: np.ndarray, alpha: float, seed: int = SEED) -> int:
    n = len(fused_train)
    plan = fold_plan(n, 5, seed)
    false_alerts = 0
    for fit_idx, query_idx in plan:
        fit_scores = fused_train[fit_idx]
        query_scores = fused_train[query_idx]
        threshold = np.quantile(fit_scores, 1.0 - alpha, method="higher")
        false_alerts += np.sum(query_scores > threshold)
    return int(false_alerts)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_exp153_inputs(datasets: list[dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    current_cache = {}
    for entry in datasets:
        ds = entry["dataset"]
        current_cache[ds] = load_current_oof_cache(ds)
    return current_cache


def _load_exp206_inputs(datasets: list[dict[str, object]]) -> dict[str, dict[str, np.ndarray]]:
    alt_cache = {}
    for entry in datasets:
        ds = entry["dataset"]
        alt_cache[ds] = load_exp206_cache(ds)
    return alt_cache


def process_dataset_206(entry: dict[str, object], cache_dir: Path) -> dict[str, object]:
    ds = str(entry["dataset"])
    X_train, X_test, target_len = load_dataset_series_only(ds)
    payload = crossfit_alternative_sources(X_train, X_test, folds=5, seed=SEED)
    
    # Save NPZ
    cache_path = cache_dir / f"{ds}.npz"
    from experiments.exp162.full_coverage_score_core import source_config_hash
    metadata = {
        "dataset": ds,
        "target_length": int(target_len),
        "source_config_hash": source_config_hash(),
        "seed": int(SEED),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "contains_test_labels": False
    }
    
    np.savez_compressed(
        cache_path,
        metadata_json=np.array(json.dumps(metadata)),
        **payload
    )
    
    return {
        "dataset_name": ds,
        "n_train": int(entry["n_train"]),
        "n_test": int(entry["n_test"]),
        "train_target_len": int(target_len),
        "abstain": 0,
        "source_names": str(list(SOURCE_NAMES)),
        "cache_file": f"{ds}.npz",
        "cache_sha256": sha256_file(cache_path),
    }


def run_stage_206(dataset_limit: int | None, output_root: Path, workers: int) -> None:
    datasets = load_population_values_only(dataset_limit)
    cache_dir = output_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Exp206 Alternative-Source OOF Reconstruction. datasets={len(datasets)} workers={workers}")
    manifest_rows = []
    errors = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(process_dataset_206, entry, cache_dir): entry["dataset"] for entry in datasets}
        for done, future in enumerate(as_completed(futures), 1):
            ds = futures[future]
            try:
                manifest_rows.append(future.result())
            except Exception as e:
                errors.append({"dataset": ds, "error": repr(e)})
            if done % 10 == 0 or done == len(futures):
                print(f"OOF CrossFit progress: {done}/{len(futures)} datasets scored. errors={len(errors)}")

    if errors:
        _write_csv(output_root / "exp206_errors.csv", errors)
        print(f"Exp206 failed with {len(errors)} errors")
        sys.exit(1)

    manifest_rows.sort(key=lambda x: x["dataset_name"])
    manifest_path = output_root / "01_score_cache_manifest.csv"
    _write_csv(manifest_path, manifest_rows)

    completion = output_root / "exp206_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": STAGE_META["206"][0], "status": "completed", "datasets": len(datasets)}, indent=2) + "\n"
    )

    print(f"Exp206 completed. Cache manifest at {manifest_path}")


def run_stage_207(dataset_limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(dataset_limit)
    print(f"Starting Exp207 Conditional Novelty Audit. datasets={len(datasets)}")

    # Load caches
    current_cache = _load_exp153_inputs(datasets)
    alt_cache = _load_exp206_inputs(datasets)

    # Fixed alpha grid
    alphas = ALPHAS
    methods = ["top2_mean", "weighted_top2", "fisher"]

    audit_rows = []
    risk_summary = []

    # Map of rule -> list of (false_alerts, n_train)
    rule_counts = defaultdict(list)

    for entry in datasets:
        ds = entry["dataset"]
        n_train = int(entry["n_train"])
        
        curr_ds = current_cache[ds]
        alt_ds = alt_cache[ds]

        for method in methods:
            fused_train, _ = compute_fused_train_test(alt_ds, method)
            for alpha in alphas:
                fa = compute_oof_false_alerts(fused_train, alpha)
                rule_counts[(method, alpha)].append((fa, n_train))
                audit_rows.append({
                    "dataset": ds,
                    "method": method,
                    "alpha": alpha,
                    "n_train": n_train,
                    "false_alerts": fa,
                    "false_alert_rate": fa / n_train,
                    "wilson_upper": wilson_upper(fa, n_train),
                })

    # Classify rules globally
    frozen_rules = {}
    for (method, alpha), counts in rule_counts.items():
        total_fa = sum(c[0] for c in counts)
        total_n = sum(c[1] for c in counts)
        global_upper = wilson_upper(total_fa, total_n)
        classification = classify_risk(global_upper)
        frozen_rules[f"{method}__{alpha}"] = classification
        
        risk_summary.append({
            "method": method,
            "alpha": alpha,
            "total_false_alerts": total_fa,
            "total_runs": total_n,
            "false_alert_rate": total_fa / total_n,
            "wilson_upper_bound": global_upper,
            "classification": classification,
        })

    # Save frozen rules config
    rules_path = output_root / "exp207_frozen_rules.json"
    rules_path.write_text(json.dumps(frozen_rules, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Save CSVs
    _write_csv(output_root / "exp207_dataset_audit.csv", audit_rows)
    _write_csv(output_root / "exp207_summary.csv", risk_summary)

    completion = output_root / "exp207_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": STAGE_META["207"][0], "status": "completed", "datasets": len(datasets)}, indent=2) + "\n"
    )
    print(f"Exp207 completed. Rules frozen at {rules_path}")


def run_stage_208(dataset_limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(dataset_limit)
    print(f"Starting Exp208 Continuous Alternative-Source Fusion. datasets={len(datasets)}")

    # Load frozen rules
    rules_path = output_root / "exp207_frozen_rules.json"
    if not rules_path.exists():
        raise FileNotFoundError(f"Exp207 frozen rules not found at {rules_path}")
    frozen_rules = json.loads(rules_path.read_text(encoding="utf-8"))

    current_cache = _load_exp153_inputs(datasets)
    alt_cache = _load_exp206_inputs(datasets)

    methods = ["top2_mean", "weighted_top2", "fisher"]
    alphas = ALPHAS

    prediction_rows = []

    # 1. Label-free prediction
    for entry in datasets:
        ds = str(entry["dataset"])
        n_train = int(entry["n_train"])
        n_test = int(entry["n_test"])
        excluded = set(entry["hard_indices"]) | set(entry["standard_indices"]) | set(entry["priority_indices"])

        curr_ds = current_cache[ds]
        alt_ds = alt_cache[ds]

        for method in methods:
            fused_train, fused_test = compute_fused_train_test(alt_ds, method)
            for alpha in alphas:
                threshold = np.quantile(fused_train, 1.0 - alpha, method="higher")
                test_candidates = set(np.where(fused_test > threshold)[0].tolist())
                final_candidates = test_candidates - excluded

                rule_key = f"{method}__{alpha}"
                lane = frozen_rules[rule_key]

                prediction_rows.append({
                    "experiment_id": STAGE_META["208"][0],
                    "dataset": ds,
                    "variant": f"{method}__{alpha}",
                    "size_stratum": "100-199" if n_train < 200 else "200+",
                    "n_train": n_train,
                    "n_test": n_test,
                    "alpha": alpha,
                    "candidate_indices": sorted(list(final_candidates)),
                    "candidate_count": len(final_candidates),
                    "final_lane": lane,
                    "autonomous_alert": False,
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "uses_family_name": 0,
                    "uses_dataset_name_for_policy": 0,
                    "uses_test_length_for_policy": 0,
                    "uses_zero_f1_membership": 0,
                    "retrospective_counterfactual": 1,
                    "prospective_validated": 0,
                })

    prediction_rows.sort(key=lambda x: (x["dataset"], x["variant"]))

    # Save and freeze prediction jsonl
    input_hashes = {str(rules_path): sha256_file(rules_path)}
    pred_path, manifest_path = freeze_predictions(
        "208", STAGE_META["208"][0], prediction_rows, output_root, input_hashes
    )

    # 2. Delayed evaluation
    results = []
    for pred in prediction_rows:
        ds = pred["dataset"]
        labels = load_test_labels_for_evaluation(ds)
        if len(labels) != pred["n_test"]:
            raise ValueError(f"TEST label size mismatch for {ds}")

        candidates = set(pred["candidate_indices"])
        baseline_hard = set(datasets[next(i for i, x in enumerate(datasets) if x["dataset"] == ds)]["hard_indices"])

        from experiments.exp201_205.core import binary_metrics
        added = binary_metrics(labels, candidates)
        baseline = binary_metrics(labels, baseline_hard)
        combined = binary_metrics(labels, baseline_hard | candidates)

        results.append({
            "experiment_id": STAGE_META["208"][0],
            "dataset_name": ds,
            "variant": pred["variant"],
            "size_stratum": pred["size_stratum"],
            "n_train": pred["n_train"],
            "n_test": pred["n_test"],
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
            "metric_scope": pred["final_lane"],
            "autonomous_exp137_unchanged": 1,
            "selected_for_operation": 0,
        })

    _write_csv(output_root / f"{STAGE_META['208'][0]}_results.csv", results)

    # Summaries
    from experiments.exp201_205.run_experiment_201_205 import _summarize
    summaries = _summarize(results)
    _write_csv(output_root / f"{STAGE_META['208'][0]}_summary.csv", summaries)

    completion = output_root / "exp208_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": STAGE_META["208"][0], "status": "completed", "datasets": len(datasets)}, indent=2) + "\n"
    )

    write_manifest(
        output_root, STAGE_META["208"][0],
        [pred_path, manifest_path, output_root / f"{STAGE_META['208'][0]}_results.csv", output_root / f"{STAGE_META['208'][0]}_summary.csv", completion],
        input_hashes,
    )
    print(f"Exp208 completed.")


def run_stage_209(dataset_limit: int | None, output_root: Path) -> None:
    datasets = load_population_values_only(dataset_limit)
    print(f"Starting Exp209 Seven-Source Joint Normal Novelty. datasets={len(datasets)}")

    current_cache = _load_exp153_inputs(datasets)
    alt_cache = _load_exp206_inputs(datasets)

    methods = ["rank_gaussian_diagonal", "shrinkage_covariance"]
    alphas = ALPHAS

    # 1. We first perform a mini OOF audit to classify joint rules under normal conditions
    rule_counts = defaultdict(list)
    for entry in datasets:
        ds = entry["dataset"]
        n_train = int(entry["n_train"])
        curr_ds = current_cache[ds]
        alt_ds = alt_cache[ds]

        for method in methods:
            res = compute_joint_train_test(curr_ds, alt_ds, method)
            if res is None:
                continue
            fused_train, _ = res
            for alpha in alphas:
                fa = compute_oof_false_alerts(fused_train, alpha)
                rule_counts[(method, alpha)].append((fa, n_train))

    frozen_rules = {}
    for (method, alpha), counts in rule_counts.items():
        total_fa = sum(c[0] for c in counts)
        total_n = sum(c[1] for c in counts)
        global_upper = wilson_upper(total_fa, total_n)
        frozen_rules[f"{method}__{alpha}"] = classify_risk(global_upper)

    # 2. Label-free prediction
    prediction_rows = []
    for entry in datasets:
        ds = str(entry["dataset"])
        n_train = int(entry["n_train"])
        n_test = int(entry["n_test"])
        excluded = set(entry["hard_indices"]) | set(entry["standard_indices"]) | set(entry["priority_indices"])

        curr_ds = current_cache[ds]
        alt_ds = alt_cache[ds]

        for method in methods:
            res = compute_joint_train_test(curr_ds, alt_ds, method)
            for alpha in alphas:
                rule_key = f"{method}__{alpha}"
                lane = frozen_rules.get(rule_key, "research_sensitivity_only")

                if res is None:
                    # Abstained prediction row
                    prediction_rows.append({
                        "experiment_id": STAGE_META["209"][0],
                        "dataset": ds,
                        "variant": rule_key,
                        "size_stratum": "100-199" if n_train < 200 else "200+",
                        "n_train": n_train,
                        "n_test": n_test,
                        "alpha": alpha,
                        "candidate_indices": [],
                        "candidate_count": 0,
                        "final_lane": lane,
                        "autonomous_alert": False,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                        "uses_family_name": 0,
                        "uses_dataset_name_for_policy": 0,
                        "uses_test_length_for_policy": 0,
                        "uses_zero_f1_membership": 0,
                        "retrospective_counterfactual": 1,
                        "prospective_validated": 0,
                        "abstained": 1,
                    })
                else:
                    fused_train, fused_test = res
                    threshold = np.quantile(fused_train, 1.0 - alpha, method="higher")
                    test_candidates = set(np.where(fused_test > threshold)[0].tolist())
                    final_candidates = test_candidates - excluded

                    prediction_rows.append({
                        "experiment_id": STAGE_META["209"][0],
                        "dataset": ds,
                        "variant": rule_key,
                        "size_stratum": "100-199" if n_train < 200 else "200+",
                        "n_train": n_train,
                        "n_test": n_test,
                        "alpha": alpha,
                        "candidate_indices": sorted(list(final_candidates)),
                        "candidate_count": len(final_candidates),
                        "final_lane": lane,
                        "autonomous_alert": False,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                        "uses_family_name": 0,
                        "uses_dataset_name_for_policy": 0,
                        "uses_test_length_for_policy": 0,
                        "uses_zero_f1_membership": 0,
                        "retrospective_counterfactual": 1,
                        "prospective_validated": 0,
                        "abstained": 0,
                    })

    prediction_rows.sort(key=lambda x: (x["dataset"], x["variant"]))

    # Save and freeze prediction jsonl
    pred_path, manifest_path = freeze_predictions(
        "209", STAGE_META["209"][0], prediction_rows, output_root, {}
    )

    # 3. Delayed evaluation
    results = []
    for pred in prediction_rows:
        ds = pred["dataset"]
        labels = load_test_labels_for_evaluation(ds)
        if len(labels) != pred["n_test"]:
            raise ValueError(f"TEST label size mismatch for {ds}")

        candidates = set(pred["candidate_indices"])
        baseline_hard = set(datasets[next(i for i, x in enumerate(datasets) if x["dataset"] == ds)]["hard_indices"])

        from experiments.exp201_205.core import binary_metrics
        added = binary_metrics(labels, candidates)
        baseline = binary_metrics(labels, baseline_hard)
        combined = binary_metrics(labels, baseline_hard | candidates)

        results.append({
            "experiment_id": STAGE_META["209"][0],
            "dataset_name": ds,
            "variant": pred["variant"],
            "size_stratum": pred["size_stratum"],
            "n_train": pred["n_train"],
            "n_test": pred["n_test"],
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
            "metric_scope": pred["final_lane"],
            "autonomous_exp137_unchanged": 1,
            "selected_for_operation": 0,
        })

    _write_csv(output_root / f"{STAGE_META['209'][0]}_results.csv", results)

    # Summaries
    from experiments.exp201_205.run_experiment_201_205 import _summarize
    summaries = _summarize(results)
    _write_csv(output_root / f"{STAGE_META['209'][0]}_summary.csv", summaries)

    completion = output_root / "exp209_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": STAGE_META["209"][0], "status": "completed", "datasets": len(datasets)}, indent=2) + "\n"
    )

    write_manifest(
        output_root, STAGE_META["209"][0],
        [pred_path, manifest_path, output_root / f"{STAGE_META['209'][0]}_results.csv", output_root / f"{STAGE_META['209'][0]}_summary.csv", completion],
        {},
    )
    print(f"Exp209 completed.")


def run_stage_210(dataset_limit: int | None, output_root: Path) -> None:
    print(f"Starting Exp210 Independent Frontier Reaggregation.")
    stage_files = {
        "208": output_root / f"{STAGE_META['208'][0]}_results.csv",
        "209": output_root / f"{STAGE_META['209'][0]}_results.csv",
    }
    detail = output_root / f"{STAGE_META['210'][0]}_results.csv"
    summary = output_root / f"{STAGE_META['210'][0]}_summary.csv"
    
    independently_reaggregate(
        output_root=output_root,
        stage_files=stage_files,
        result_path=detail,
        summary_path=summary,
        bootstrap_iterations=1000,
    )
    
    completion = output_root / "exp210_completion.json"
    completion.write_text(
        json.dumps({"experiment_id": STAGE_META["210"][0], "status": "completed", "datasets": 251}, indent=2) + "\n"
    )
    
    write_manifest(
        output_root, STAGE_META["210"][0],
        [detail, summary, completion],
        {}
    )
    print("Exp210 independent reaggregation completed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp206-210 runner")
    parser.add_argument("--stage", required=True, choices=list(STAGE_META))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=DATA_DIR / "exp206_210_residual_coverage")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.stage == "206":
        run_stage_206(args.dataset_limit, args.output_root, args.workers)
    elif args.stage == "207":
        run_stage_207(args.dataset_limit, args.output_root)
    elif args.stage == "208":
        run_stage_208(args.dataset_limit, args.output_root)
    elif args.stage == "209":
        run_stage_209(args.dataset_limit, args.output_root)
    elif args.stage == "210":
        run_stage_210(args.dataset_limit, args.output_root)


if __name__ == "__main__":
    main()
