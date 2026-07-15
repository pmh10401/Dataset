from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp186_200.core import (
    SEED,
    agreement_tiers,
    apply_count_cap,
    conformal_count_cap,
    stable_dataset_guard,
    validate_disjoint_indices,
    empirical_upper_tail_pvalues,
)
from experiments.exp186_200.artifacts import (
    DATA_DIR,
    FrozenInputs,
    load_frozen_inputs,
    freeze_predictions,
    load_verified_predictions,
    load_test_labels_for_evaluation,
    verify_file_hash,
)

DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DEFAULT_OUTPUT_ROOT = DATA_DIR / "exp186_200_unattended"
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
RESULTS_DIR = DATA_DIR
EXPECTED_DATASETS = 639
EXISTING_SCORE_CACHE = Path(
    "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements/cache"
)
CANONICAL_SCORE_CACHE = DATA_DIR / "exp177_185_unattended/canonical_score_cache"

STAGE_META = {
    "186": ("experiment_186_conservative_canonical_knn_agreement", EXPECTED_DATASETS),
    "187": ("experiment_187_conservative_dual_canonical_agreement", EXPECTED_DATASETS),
    "188": ("experiment_188_train_oof_stability_guarded_review", EXPECTED_DATASETS),
    "189": ("experiment_189_train_normal_workload_guarded_review", EXPECTED_DATASETS),
    "190": ("experiment_190_three_tier_review_routing", EXPECTED_DATASETS * 3),
    "191": ("experiment_191_conservative_canonical_shadow_agreement", EXPECTED_DATASETS),
    "192": ("experiment_192_conservative_dual_backend_shadow_agreement", EXPECTED_DATASETS),
    "193": ("experiment_193_three_existing_source_canonical_confirmation", EXPECTED_DATASETS),
    "194": ("experiment_194_fold_stable_shadow_agreement", EXPECTED_DATASETS),
    "195": ("experiment_195_zero_normal_candidate_shadow_guard", EXPECTED_DATASETS),
    "196": ("experiment_196_expanded_train_risk_review", EXPECTED_DATASETS),
    "197": ("experiment_197_conservative_canonical_unique_union", EXPECTED_DATASETS),
    "198": ("experiment_198_existing_one_canonical_alpha_001", EXPECTED_DATASETS),
    "199": ("experiment_199_expanded_two_of_six_train_workload_guard", EXPECTED_DATASETS),
    "200": ("experiment_200_frozen_frontier_reaggregation", EXPECTED_DATASETS * 3),
}

STAGE_EXPECTED_ROWS = {
    "186": 639,
    "187": 639,
    "188": 639,
    "189": 639,
    "190": 1917,
    "191": 639,
    "192": 639,
    "193": 639,
    "194": 639,
    "195": 639,
    "196": 639,
    "197": 639,
    "198": 639,
    "199": 639,
    "200": 1917,
}


def get_conservative_alpha(policy: dict[str, object]) -> float:
    for variant in policy.get("policy_variants", []):
        if variant.get("risk_target") == 0.005:
            return float(variant["selected_alpha"])
    raise ValueError("Frozen Exp177 policy has no risk_target=0.005 alpha")


def dataset_count_cap(precompute: dict[str, object], stage: str) -> int:
    """Return a dataset-local cap from its cross-fitted TRAIN-normal fold counts."""
    key = f"candidate_counts_{stage}"
    counts = [int(value) for value in precompute[key]]
    return conformal_count_cap(counts, alpha=0.01)


def _dataset_precompute_worker(
    dataset: str,
    alpha_cons: float,
    manifest_153_157_row: dict[str, str],
    canonical_manifest_row: dict[str, str]
) -> dict[str, object]:
    existing_path = EXISTING_SCORE_CACHE / manifest_153_157_row["cache_file"]
    verify_file_hash(existing_path, manifest_153_157_row["cache_sha256"], "Exp153 score cache")
    from experiments.exp177_185.core import SOURCE_CACHE_KEYS
    with np.load(existing_path, allow_pickle=False) as archive:
        existing_sources_train = {
            name: np.asarray(archive[f"{key}__train_oof"], dtype=float)
            for name, key in SOURCE_CACHE_KEYS.items()
        }

    canonical_path = CANONICAL_SCORE_CACHE / canonical_manifest_row["score_file"]
    verify_file_hash(canonical_path, canonical_manifest_row["score_sha256"], "canonical score cache")
    with np.load(canonical_path, allow_pickle=False) as archive:
        canonical_knn_train = np.asarray(archive["knn_local_gap__train_oof"], dtype=float)
        canonical_robust_train = np.asarray(archive["robust_diagonal__train_oof"], dtype=float)

    n_train = len(next(iter(existing_sources_train.values())))
    from experiments.exp177_185.core import _fold_plan
    folds = _fold_plan(n_train, seed=SEED)

    fold_results = []
    candidate_counts_189 = []
    candidate_counts_199 = []
    candidate_counts_195 = []

    for fit, query in folds:
        query_alerts_189 = 0
        query_alerts_199 = 0
        query_alerts_195 = 0
        for i in range(len(query)):
            idx_in_train = query[i]

            support_cons = 0
            support_01 = 0
            for name in SOURCE_CACHE_KEYS:
                val = existing_sources_train[name][idx_in_train]
                pval = empirical_upper_tail_pvalues(existing_sources_train[name][fit], np.array([val]))[0]
                if pval <= alpha_cons:
                    support_cons += 1
                if pval <= 0.01:
                    support_01 += 1

            knn_val = canonical_knn_train[idx_in_train]
            knn_pval = empirical_upper_tail_pvalues(canonical_knn_train[fit], np.array([knn_val]))[0]

            robust_val = canonical_robust_train[idx_in_train]
            robust_pval = empirical_upper_tail_pvalues(canonical_robust_train[fit], np.array([robust_val]))[0]

            if support_cons >= 2:
                query_alerts_189 += 1

            support_six = support_01
            if knn_pval <= 0.01:
                support_six += 1
            if robust_pval <= 0.01:
                support_six += 1
            if support_six >= 2:
                query_alerts_199 += 1

            if support_cons >= 2 and knn_pval <= 0.005:
                query_alerts_195 += 1

        fold_results.append(query_alerts_189 == 0)
        candidate_counts_189.append(query_alerts_189)
        candidate_counts_199.append(query_alerts_199)
        candidate_counts_195.append(query_alerts_195)

    return {
        "dataset": dataset,
        "fold_results": fold_results,
        "candidate_counts_189": candidate_counts_189,
        "candidate_counts_199": candidate_counts_199,
        "candidate_counts_195": candidate_counts_195,
    }


def predict_dataset(
    stage: str,
    inputs: FrozenInputs,
    dataset: str,
    precompute_results: dict[str, dict],
    alpha_cons: float,
    canonical_manifest: dict,
    cap_value_189: int = 0,
    cap_value_199: int = 0,
) -> list[dict[str, object]]:
    from experiments.exp177_185.core import SOURCE_CACHE_KEYS

    exp137_row = next(row for row in inputs.exp137_rows if row["dataset_name"] == dataset)

    def parse_indices(value: object) -> set[int]:
        text = str(value or "").replace(",", " ").strip()
        return {int(token) for token in text.split()} if text else set()

    excluded = (
        parse_indices(exp137_row["hard_alert_indices"])
        | parse_indices(exp137_row["standard_review_indices"])
        | parse_indices(exp137_row["priority_review_indices"])
    )

    manifest_153_157 = {row["dataset_name"]: row for row in inputs.exp153_157_manifest}
    existing_row = manifest_153_157[dataset]
    existing_path = EXISTING_SCORE_CACHE / existing_row["cache_file"]
    verify_file_hash(existing_path, existing_row["cache_sha256"], "Exp153 score cache")
    with np.load(existing_path, allow_pickle=False) as archive:
        n_test = int(json.loads(str(archive["metadata_json"].item()))["n_test"])
        existing_sources = {
            name: {
                "train": np.asarray(archive[f"{key}__train_oof"], dtype=float),
                "test": np.asarray(archive[f"{key}__test"], dtype=float),
            }
            for name, key in SOURCE_CACHE_KEYS.items()
        }

    existing_p = {
        name: empirical_upper_tail_pvalues(
            existing_sources[name]["train"], existing_sources[name]["test"]
        )
        for name in SOURCE_CACHE_KEYS
    }

    canonical_row = canonical_manifest[dataset]
    canonical_path = CANONICAL_SCORE_CACHE / canonical_row["score_file"]
    verify_file_hash(canonical_path, canonical_row["score_sha256"], "canonical score cache")
    with np.load(canonical_path, allow_pickle=False) as archive:
        canonical_knn_train = np.asarray(archive["knn_local_gap__train_oof"], dtype=float)
        canonical_knn_test = np.asarray(archive["knn_local_gap__test"], dtype=float)
        canonical_robust_train = np.asarray(archive["robust_diagonal__train_oof"], dtype=float)
        canonical_robust_test = np.asarray(archive["robust_diagonal__test"], dtype=float)

    canonical_knn_p = empirical_upper_tail_pvalues(canonical_knn_train, canonical_knn_test)
    canonical_robust_p = empirical_upper_tail_pvalues(canonical_robust_train, canonical_robust_test)

    canonical_knn_005 = {idx for idx, p in enumerate(canonical_knn_p) if p <= 0.005} - excluded
    canonical_robust_005 = {idx for idx, p in enumerate(canonical_robust_p) if p <= 0.005} - excluded

    exp178_conservative = set()
    exp178_expanded = set()
    for p in inputs.exp178_payloads:
        if p["dataset"] == dataset:
            if p["variant"] == "risk_target_0.005":
                exp178_conservative = set(p["candidate_indices"])
            elif p["variant"] == "risk_target_0.02":
                exp178_expanded = set(p["candidate_indices"])

    precompute = precompute_results[dataset]
    fold_results = precompute["fold_results"]

    evidence_by_index = {}
    for idx in range(n_test):
        evidence_by_index[idx] = tuple(sorted([
            existing_p[name][idx] for name in SOURCE_CACHE_KEYS
        ]))

    payloads = []
    experiment_id = STAGE_META[stage][0]

    if stage == "186":
        candidates = exp178_conservative & canonical_knn_005
        payloads.append({
            "experiment_id": experiment_id, "variant": "conservative_and_canonical_knn_005",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "high_confidence_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "187":
        candidates = exp178_conservative & canonical_knn_005 & canonical_robust_005
        payloads.append({
            "experiment_id": experiment_id, "variant": "conservative_and_dual_canonical_005",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "high_confidence_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "188":
        stable = stable_dataset_guard(fold_results)
        candidates = exp178_conservative if stable else set()
        payloads.append({
            "experiment_id": experiment_id, "variant": "conservative_with_4of5_train_stability",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "standard_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "189":
        candidates = apply_count_cap(exp178_conservative, cap_value_189, evidence_by_index)
        payloads.append({
            "experiment_id": experiment_id, "variant": "conservative_with_train_conformal_count_cap",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "standard_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "train_count_cap_raw": cap_value_189,
            "train_count_cap_minimum_one": max(1, cap_value_189),
            "train_count_cap_primary": "raw",
        })
    elif stage == "190":
        tiers = agreement_tiers(exp178_conservative, canonical_knn_005)
        payloads.append({
            "experiment_id": experiment_id, "variant": "three_tier_high_confidence",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(tiers.high), "final_lane": "high_confidence_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
        payloads.append({
            "experiment_id": experiment_id, "variant": "three_tier_standard",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(tiers.standard), "final_lane": "standard_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
        payloads.append({
            "experiment_id": experiment_id, "variant": "three_tier_exploratory",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(tiers.exploratory), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "191":
        candidates = exp178_conservative & canonical_knn_005
        payloads.append({
            "experiment_id": experiment_id, "variant": "exp186_shadow",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "shadow_hard_diagnostic",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "autonomous_alert": False,
        })
    elif stage == "192":
        candidates = exp178_conservative & canonical_knn_005 & canonical_robust_005
        payloads.append({
            "experiment_id": experiment_id, "variant": "exp187_shadow",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "shadow_hard_diagnostic",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "autonomous_alert": False,
        })
    elif stage == "193":
        existing_support_at_alpha = np.sum(np.vstack([existing_p[name] <= alpha_cons for name in SOURCE_CACHE_KEYS]), axis=0)
        candidates = {idx for idx in range(n_test) if existing_support_at_alpha[idx] >= 3 and canonical_knn_p[idx] <= 0.005} - excluded
        payloads.append({
            "experiment_id": experiment_id, "variant": "existing3_plus_canonical_knn_005",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "shadow_hard_diagnostic",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "autonomous_alert": False,
        })
    elif stage == "194":
        stable = stable_dataset_guard(fold_results)
        candidates = (exp178_conservative & canonical_knn_005) if stable else set()
        payloads.append({
            "experiment_id": experiment_id, "variant": "exp191_plus_4of5_train_stability",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "shadow_hard_diagnostic",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "autonomous_alert": False,
        })
    elif stage == "195":
        zero_train = sum(precompute["candidate_counts_195"]) == 0
        candidates = (exp178_conservative & canonical_knn_005) if zero_train else set()
        payloads.append({
            "experiment_id": experiment_id, "variant": "exp191_plus_zero_train_candidate_guard",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "shadow_hard_diagnostic",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "autonomous_alert": False,
        })
    elif stage == "196":
        payloads.append({
            "experiment_id": experiment_id, "variant": "risk_target_0.02",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(exp178_expanded), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "197":
        candidates = exp178_conservative | canonical_knn_005
        evidence_flags = {}
        for idx in candidates:
            if idx in exp178_conservative and idx in canonical_knn_005:
                origin = "both"
            elif idx in exp178_conservative:
                origin = "conservative"
            else:
                origin = "canonical_knn"
            evidence_flags[str(idx)] = origin
        payloads.append({
            "experiment_id": experiment_id, "variant": "conservative_plus_canonical_unique_union",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "train_count_cap_raw": cap_value_199,
            "train_count_cap_minimum_one": max(1, cap_value_199),
            "train_count_cap_primary": "raw",
            "evidence_flags": evidence_flags,
        })
    elif stage == "198":
        existing_support_at_01 = np.sum(np.vstack([existing_p[name] <= 0.01 for name in SOURCE_CACHE_KEYS]), axis=0)
        canonical_knn_p_01 = canonical_knn_p <= 0.01
        candidates = {idx for idx in range(n_test) if existing_support_at_01[idx] >= 1 and canonical_knn_p_01[idx]} - excluded
        payloads.append({
            "experiment_id": experiment_id, "variant": "existing1_plus_canonical_knn_alpha_0.01",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "199":
        existing_support_at_01 = np.vstack([existing_p[name] <= 0.01 for name in SOURCE_CACHE_KEYS])
        c_knn_01 = canonical_knn_p <= 0.01
        c_robust_01 = canonical_robust_p <= 0.01
        support_six = np.sum(np.vstack([existing_support_at_01, c_knn_01, c_robust_01]), axis=0)
        raw_candidates = {idx for idx in range(n_test) if support_six[idx] >= 2} - excluded

        evidence_by_index_six = {}
        for idx in range(n_test):
            evidence_by_index_six[idx] = tuple(sorted([
                existing_p[name][idx] for name in SOURCE_CACHE_KEYS
            ] + [canonical_knn_p[idx], canonical_robust_p[idx]]))

        candidates = apply_count_cap(raw_candidates, cap_value_199, evidence_by_index_six)
        payloads.append({
            "experiment_id": experiment_id, "variant": "expanded_2of6_with_train_workload_guard",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(candidates), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    elif stage == "200":
        exp186_c = exp178_conservative & canonical_knn_005
        stable = stable_dataset_guard(fold_results)
        exp188_c = exp178_conservative if stable else set()
        exp197_c = exp178_conservative | canonical_knn_005
        payloads.append({
            "experiment_id": experiment_id, "variant": "reaggregate_conservative",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(exp186_c), "final_lane": "high_confidence_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
        payloads.append({
            "experiment_id": experiment_id, "variant": "reaggregate_balanced",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(exp188_c), "final_lane": "standard_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
        payloads.append({
            "experiment_id": experiment_id, "variant": "reaggregate_expanded",
            "dataset": dataset, "n_test": n_test,
            "candidate_indices": sorted(exp197_c), "final_lane": "exploratory_review",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })

    return payloads


def load_score_manifest_local(filename: str) -> dict[str, dict[str, str]]:
    path = DATA_DIR / "exp177_185_unattended" / filename
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_dataset = {row["dataset_name"]: row for row in rows}
    if len(by_dataset) != len(rows):
        raise ValueError(f"Duplicate dataset rows in {path}")
    return by_dataset


def _precompute_signature(
    datasets: list[str],
    alpha_cons: float,
    existing_manifest: dict[str, dict[str, str]],
    canonical_manifest: dict[str, dict[str, str]],
) -> str:
    contract = {
        "datasets": datasets,
        "alpha_cons": alpha_cons,
        "seed": SEED,
        "existing": {d: existing_manifest[d]["cache_sha256"] for d in datasets},
        "canonical": {d: canonical_manifest[d]["score_sha256"] for d in datasets},
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_build_precompute(
    datasets: list[str],
    alpha_cons: float,
    existing_manifest: dict[str, dict[str, str]],
    canonical_manifest: dict[str, dict[str, str]],
    output_root: Path,
    workers: int,
) -> dict[str, dict[str, object]]:
    signature = _precompute_signature(datasets, alpha_cons, existing_manifest, canonical_manifest)
    cache_path = output_root / "train_guard_precompute.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("signature") == signature and sorted(cached.get("results", {})) == datasets:
            print(f"Reusing verified TRAIN-only precompute cache: {cache_path}", flush=True)
            return cached["results"]

    results: dict[str, dict[str, object]] = {}
    print(f"Building TRAIN-only precompute cache with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _dataset_precompute_worker,
                dataset,
                alpha_cons,
                existing_manifest[dataset],
                canonical_manifest[dataset],
            ): dataset
            for dataset in datasets
        }
        for done, future in enumerate(as_completed(futures), start=1):
            dataset = futures[future]
            results[dataset] = future.result()
            print(
                f"Progress: [ {done} / {len(datasets)} ] rows={done} dataset={dataset} errors=0",
                flush=True,
            )

    output_root.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(
            {"signature": signature, "seed": SEED, "results": results},
            ensure_ascii=False,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(cache_path)
    return results


def run_prediction_stage(stage: str, dataset_limit: int | None = None) -> None:
    inputs = load_frozen_inputs()
    alpha_cons = get_conservative_alpha(inputs.exp177_policy)
    canonical_manifest = load_score_manifest_local("exp180_canonical_score_manifest.csv")

    # Precomputation
    workers = int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7"))
    eligible_datasets = sorted(
        [row["dataset_name"] for row in inputs.exp137_rows if int(row["train_normal_count"]) >= 20]
    )
    if len(eligible_datasets) != EXPECTED_DATASETS:
        raise ValueError(f"Eligible coverage changed: {len(eligible_datasets)} != {EXPECTED_DATASETS}")
    if dataset_limit is not None:
        eligible_datasets = eligible_datasets[:dataset_limit]

    manifest_153_157 = {row["dataset_name"]: row for row in inputs.exp153_157_manifest}
    missing_existing = sorted(set(eligible_datasets) - set(manifest_153_157))
    missing_canonical = sorted(set(eligible_datasets) - set(canonical_manifest))
    if missing_existing or missing_canonical:
        raise ValueError(
            f"Score coverage incomplete: existing={missing_existing[:5]} canonical={missing_canonical[:5]}"
        )
    precompute_results = load_or_build_precompute(
        eligible_datasets,
        alpha_cons,
        manifest_153_157,
        canonical_manifest,
        OUTPUT_ROOT,
        workers,
    )

    payloads = []
    print(f"Generating prediction payloads for stage {stage}...", flush=True)
    for done, dataset in enumerate(eligible_datasets, start=1):
        cap_189 = dataset_count_cap(precompute_results[dataset], "189")
        cap_199 = dataset_count_cap(precompute_results[dataset], "199")
        res = predict_dataset(
            stage=stage,
            inputs=inputs,
            dataset=dataset,
            precompute_results=precompute_results,
            alpha_cons=alpha_cons,
            canonical_manifest=canonical_manifest,
            cap_value_189=cap_189,
            cap_value_199=cap_199,
        )
        payloads.extend(res)
        print(
            f"Progress: [ {done} / {len(eligible_datasets)} ] rows={len(payloads)} dataset={dataset} errors=0",
            flush=True,
        )

    print(f"Freezing predictions for stage {stage}...", flush=True)
    freeze_predictions(stage, payloads, OUTPUT_ROOT, STAGE_META[stage][0])

    # Write prediction-only placeholder result/summary CSVs
    # to follow the exact pattern from exp177_185
    detail_rows = []
    for p in payloads:
        detail_rows.append({
            "experiment_id": STAGE_META[stage][0],
            "dataset_name": p["dataset"],
            "config_name": p["variant"],
            "candidate_count": len(p["candidate_indices"]),
            "candidate_indices": " ".join(str(i) for i in p["candidate_indices"]),
            "mean_f1": "",
            "metric_scope": "prediction_only_no_test_labels",
        })
    detail_path = RESULTS_DIR / f"{STAGE_META[stage][0]}_results.csv"
    summary_path = RESULTS_DIR / f"{STAGE_META[stage][0]}_summary.csv"

    # Write detail
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name", "config_name", "candidate_count", "candidate_indices", "mean_f1", "metric_scope"])
        writer.writeheader()
        writer.writerows(detail_rows)

    # Write summary
    summary_rows = [{
        "experiment_id": STAGE_META[stage][0],
        "config_name": "frozen_prediction",
        "datasets": len({p["dataset"] for p in payloads}),
        "rows": len(payloads),
        "mean_f1": "",
        "metric_scope": "prediction_only_no_test_labels",
        "prediction_sha256": core_sha256(OUTPUT_ROOT / f"exp{stage}_frozen_predictions.jsonl"),
    }]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "config_name", "datasets", "rows", "mean_f1", "metric_scope", "prediction_sha256"])
        writer.writeheader()
        writer.writerows(summary_rows)



def core_sha256(path: Path) -> str:
    from experiments.exp186_200.core import sha256_file
    return sha256_file(path)


def evaluate_stage(stage: str) -> None:
    """This function is evaluation-only and reads test labels after prediction payloads are frozen."""
    print(f"Evaluating stage {stage} against TEST labels...", flush=True)
    inputs = load_frozen_inputs()
    jsonl_path = OUTPUT_ROOT / f"exp{stage}_frozen_predictions.jsonl"
    manifest_path = OUTPUT_ROOT / f"exp{stage}_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_datasets = int(manifest["dataset_count"])
    payloads = load_verified_predictions(jsonl_path, manifest_path, expected_datasets)

    # Group by variant
    by_variant = {}
    for p in payloads:
        by_variant.setdefault(p["variant"], []).append(p)

    detail_rows = []
    summary_rows = []

    # Get Exp137 hard alerts for remaining anomaly recovery
    exp137_by_dataset = {row["dataset_name"]: row for row in inputs.exp137_rows}

    def parse_indices(value: object) -> set[int]:
        text = str(value or "").replace(",", " ").strip()
        return {int(token) for token in text.split()} if text else set()

    for variant, var_payloads in by_variant.items():
        var_tps = []
        var_fps = []
        var_fns = []
        var_f1s = []
        var_counts = []
        var_recovered = []

        for p in var_payloads:
            dataset = p["dataset"]
            candidates = set(p["candidate_indices"])
            
            # Load y_test
            labels = load_test_labels_for_evaluation(dataset)
            truth = set(np.flatnonzero(labels != 0).astype(int).tolist())

            tp = len(candidates & truth)
            fp = len(candidates - truth)
            fn = len(truth - candidates)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-15, precision + recall)

            # Inputs already exclude every frozen Exp137 lane. This is recovery
            # relative to the complete frozen routing, not only its Hard lane.
            exp137_row = exp137_by_dataset[dataset]
            frozen_routing = (
                parse_indices(exp137_row["hard_alert_indices"])
                | parse_indices(exp137_row["standard_review_indices"])
                | parse_indices(exp137_row["priority_review_indices"])
            )
            remaining_truth = truth - frozen_routing
            recovered = len(candidates & remaining_truth)

            scope = (
                "retrospective_shadow_hard_diagnostic"
                if p.get("final_lane") == "shadow_hard_diagnostic"
                else "retrospective_human_assisted_review"
            )

            var_tps.append(tp)
            var_fps.append(fp)
            var_fns.append(fn)
            var_f1s.append(f1)
            var_counts.append(len(candidates))
            var_recovered.append(recovered)

            detail_rows.append({
                "experiment_id": STAGE_META[stage][0],
                "dataset_name": dataset,
                "config_name": variant,
                "candidate_count": len(candidates),
                "candidate_indices": " ".join(str(i) for i in sorted(candidates)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "recovered_after_frozen_routing": recovered,
                "metric_scope": scope,
                "retrospective": 1,
                "prospective_validated": 0,
                "selected_for_operation": 0,
            })

        summary_rows.append({
            "experiment_id": STAGE_META[stage][0],
            "config_name": variant,
            "datasets": len(var_payloads),
            "review_candidates": sum(var_counts),
            "review_tp": sum(var_tps),
            "review_fp": sum(var_fps),
            "review_precision": sum(var_tps) / max(1, sum(var_tps) + sum(var_fps)),
            "review_recall": sum(var_tps) / max(1, sum(var_tps) + sum(var_fns)),
            "mean_f1": float(np.mean(var_f1s)),
            "active_datasets": sum(count > 0 for count in var_counts),
            "zero_candidate_datasets": sum(count == 0 for count in var_counts),
            "workload_median": float(np.median(var_counts)),
            "workload_p90": float(np.percentile(var_counts, 90)),
            "workload_p95": float(np.percentile(var_counts, 95)),
            "workload_max": max(var_counts),
            "recovered_after_frozen_routing_total": sum(var_recovered),
            "metric_scope": (
                "retrospective_shadow_hard_diagnostic"
                if var_payloads[0].get("final_lane") == "shadow_hard_diagnostic"
                else "retrospective_human_assisted_review"
            ),
            "retrospective": 1,
            "prospective_validated": 0,
            "selected_for_operation": 0,
        })

    # Write evaluation outputs
    detail_path = RESULTS_DIR / f"{STAGE_META[stage][0]}_results.csv"
    summary_path = RESULTS_DIR / f"{STAGE_META[stage][0]}_summary.csv"

    # Write detail
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "dataset_name", "config_name", "candidate_count", "candidate_indices", "tp", "fp", "fn", "precision", "recall", "f1", "recovered_after_frozen_routing", "metric_scope", "retrospective", "prospective_validated", "selected_for_operation"])
        writer.writeheader()
        writer.writerows(detail_rows)

    # Write summary
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "config_name", "datasets", "review_candidates", "review_tp", "review_fp", "review_precision", "review_recall", "mean_f1", "active_datasets", "zero_candidate_datasets", "workload_median", "workload_p90", "workload_p95", "workload_max", "recovered_after_frozen_routing_total", "metric_scope", "retrospective", "prospective_validated", "selected_for_operation"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Evaluation complete for stage {stage}.", flush=True)


def run_reaggregation() -> dict[str, object]:
    from experiments.exp186_200.independent_reaggregate import run_reaggregation as implementation

    return implementation(results_dir=RESULTS_DIR, output_root=OUTPUT_ROOT)


def write_completion(stage: str) -> None:
    prediction_path = OUTPUT_ROOT / f"exp{stage}_frozen_predictions.jsonl"
    manifest_path = OUTPUT_ROOT / f"exp{stage}_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion_payload = {
        "stage": stage,
        "experiment_id": STAGE_META[stage][0],
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrospective": True,
        "prospective_validated": False,
        "selected_for_operation": False,
        "prediction_sha256": core_sha256(prediction_path),
        "prediction_rows": manifest["prediction_rows"],
        "dataset_count": manifest["dataset_count"],
        "evaluation_completed": True,
        "reaggregation_completed": stage == "200",
    }
    (OUTPUT_ROOT / f"exp{stage}_completion.json").write_text(
        json.dumps(completion_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_stage(stage: str, dataset_limit: int | None = None) -> None:
    run_prediction_stage(stage, dataset_limit=dataset_limit)
    evaluate_stage(stage)
    if stage == "200":
        run_reaggregation()
    write_completion(stage)


def main() -> None:
    global OUTPUT_ROOT, RESULTS_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=tuple(STAGE_META))
    parser.add_argument("--evaluate", action="store_true", help="Run post-hoc evaluation only (requires TEST labels)")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.output_dir is not None:
        OUTPUT_ROOT = args.output_dir.resolve()
        RESULTS_DIR = OUTPUT_ROOT
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        evaluate_stage(args.stage)
    else:
        run_stage(args.stage, dataset_limit=args.dataset_limit)



if __name__ == "__main__":
    main()
