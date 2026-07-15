#!/usr/bin/env python3
"""
Exp164 Checkpointed Runner for TRAIN-Normal Agreement Risk Audit.
"""
from __future__ import annotations

import os
# Restrict BLAS/OpenMP inner threads before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import json
import sys
import time
import hashlib
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

# Verify database and frozen population paths
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DB_EXPECTED_HASH = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"

EXP162_MANIFEST_PATH = PROJECT_DIR / "outputs" / "exp162_full_coverage_alternative_score_cache" / "02_cache_manifest.csv"
EXP162_MANIFEST_EXPECTED_HASH = "ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1"

EXP162_CONTRACT_PATH = PROJECT_DIR / "outputs" / "exp162_full_coverage_alternative_score_cache" / "01_source_contract.json"
EXP162_CONTRACT_EXPECTED_HASH = "0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461"

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def checkpoint_contract_matches(
    checkpoint: dict,
    config_hash: str,
    database_sha256: str,
    population_sha256: str,
) -> bool:
    return (
        checkpoint.get("config_hash") == config_hash
        and checkpoint.get("database_sha256") == database_sha256
        and checkpoint.get("population_sha256") == population_sha256
    )

def load_frozen_population() -> list[str]:
    # Verify exp162 files
    if not EXP162_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Exp162 manifest file not found at {EXP162_MANIFEST_PATH}")
    manifest_hash = get_sha256(EXP162_MANIFEST_PATH)
    if manifest_hash != EXP162_MANIFEST_EXPECTED_HASH:
        raise ValueError(f"Exp162 manifest hash mismatch! Expected {EXP162_MANIFEST_EXPECTED_HASH}, got {manifest_hash}")

    if not EXP162_CONTRACT_PATH.exists():
        raise FileNotFoundError(f"Exp162 contract file not found at {EXP162_CONTRACT_PATH}")
    contract_hash = get_sha256(EXP162_CONTRACT_PATH)
    if contract_hash != EXP162_CONTRACT_EXPECTED_HASH:
        raise ValueError(f"Exp162 contract hash mismatch! Expected {EXP162_CONTRACT_EXPECTED_HASH}, got {contract_hash}")

    # Verify DB
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    db_hash = get_sha256(DB_PATH)
    if db_hash != DB_EXPECTED_HASH:
        raise ValueError(f"Database SHA256 mismatch! Expected {DB_EXPECTED_HASH}, got {db_hash}")

    # Load dataset names from Exp162 manifest
    datasets = []
    with EXP162_MANIFEST_PATH.open(mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "dataset" in row:
                datasets.append(row["dataset"])
    if len(set(datasets)) != len(datasets):
        raise ValueError("Exp162 manifest contains duplicate dataset names")
    datasets = sorted(datasets)
    if len(datasets) != 1117:
        raise ValueError(f"Expected 1117 datasets in population, got {len(datasets)}")
    return datasets

def get_global_config_hash() -> str:
    from experiments.exp162.full_coverage_score_core import source_config_hash
    src_hash = source_config_hash()

    # fold configuration
    fold_config = {
        "seed": 20260715,
        "alphas": [0.005, 0.01, 0.02, 0.05],
        "variants": ["train_audit_2of3", "train_audit_3of3"],
        "risk_targets": [0.005, 0.01, 0.02],
        "wilson_confidence": 0.95,
        "min_train_for_audit": 5
    }
    fold_hash = hashlib.sha256(json.dumps(fold_config, sort_keys=True).encode("utf-8")).hexdigest()

    combined = src_hash + fold_hash
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def process_single_dataset(dataset_name: str, output_dir: Path, global_config_hash: str) -> dict:
    # Set single-thread environment variables inside child worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from experiments.exp164.exp164_train_bridge import load_train_normal_raw
    from experiments.exp164.crossfit_sources import audit_dataset
    from experiments.exp164.exp164_cache_io import write_cache_atomic

    cache_file = output_dir / f"{dataset_name}_cache.npz"
    try:
        raw_series = load_train_normal_raw(dataset_name)
        results = audit_dataset(dataset_name, raw_series)

        if results is None:
            # Abstention
            return {
                "dataset": dataset_name,
                "success": True,
                "abstained": True,
                "file_path": None,
                "file_hash": None,
                "n_train": len(raw_series),
                "error": None,
                "traceback": None,
            }

        # Format payload for cache write
        alphas = (0.005, 0.01, 0.02, 0.05)
        sources = ("spectral", "shapelet", "interval")
        n_train = results["n_train"]

        payload = {
            "query_score_spectral": results["query_scores"]["spectral"].astype(np.float64),
            "query_score_shapelet": results["query_scores"]["shapelet"].astype(np.float64),
            "query_score_interval": results["query_scores"]["interval"].astype(np.float64),
            "fold_indices": results["fold_indices"].astype(np.int32),
            "pseudo_run_indices": results["pseudo_run_indices"].astype(np.int32),
            "fit_counts": results["fit_counts"].astype(np.int32),
            "target_lengths": results["target_lengths"].astype(np.int32),
            "thresholds_spectral": np.column_stack([results["thresholds"][a]["spectral"] for a in alphas]).astype(np.float64),
            "thresholds_shapelet": np.column_stack([results["thresholds"][a]["shapelet"] for a in alphas]).astype(np.float64),
            "thresholds_interval": np.column_stack([results["thresholds"][a]["interval"] for a in alphas]).astype(np.float64),
            "flags_spectral": np.column_stack([results["flags"][a]["spectral"] for a in alphas]).astype(bool),
            "flags_shapelet": np.column_stack([results["flags"][a]["shapelet"] for a in alphas]).astype(bool),
            "flags_interval": np.column_stack([results["flags"][a]["interval"] for a in alphas]).astype(bool),
            "agreement_2of3": np.column_stack([results["agreement_2of3"][a] for a in alphas]).astype(bool),
            "agreement_3of3": np.column_stack([results["agreement_3of3"][a] for a in alphas]).astype(bool),
        }

        metadata = {
            "dataset": dataset_name,
            "n_train": n_train,
            "fold_method": results["fold_method"],
            "source_config_hash": global_config_hash,
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "alphas": list(alphas)
        }

        file_hash = write_cache_atomic(cache_file, payload, metadata)
        rel_path = os.path.relpath(cache_file, PROJECT_DIR)

        return {
            "dataset": dataset_name,
            "success": True,
            "abstained": False,
            "file_path": rel_path,
            "file_hash": file_hash,
            "n_train": n_train,
            "error": None,
            "traceback": None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        rel_path = os.path.relpath(cache_file, PROJECT_DIR) if cache_file.exists() else None
        return {
            "dataset": dataset_name,
            "success": False,
            "abstained": False,
            "file_path": rel_path,
            "file_hash": None,
            "n_train": 0,
            "error": str(e),
            "traceback": tb,
        }

def run_experiment():
    parser = argparse.ArgumentParser(description="Exp164 TRAIN-Normal Agreement Risk Audit Runner")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--workers", type=int, default=7, help="Num workers")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Reset checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/exp164_train_normal_agreement_risk_audit", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_config_hash = get_global_config_hash()
    db_sha256 = get_sha256(DB_PATH)

    checkpoint_file = output_dir / "CHECKPOINT.jsonl"
    if args.reset_checkpoint:
        print("Resetting checkpoint...")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        for f in output_dir.glob("*_cache.npz"):
            f.unlink()

    population = load_frozen_population()
    frozen_population_sha = hashlib.sha256(("\n".join(population) + "\n").encode("utf-8")).hexdigest()

    if args.dataset_limit is not None:
        population = population[:args.dataset_limit]
    run_population_sha = hashlib.sha256(("\n".join(population) + "\n").encode("utf-8")).hexdigest()

    # Read checkpoint
    checkpoints = {}
    reused_records = []
    if checkpoint_file.exists():
        with checkpoint_file.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    checkpoints[row["dataset"]] = row
                except Exception:
                    continue

    # Filter datasets
    to_process = []
    from experiments.exp164.exp164_cache_io import validate_cache
    for ds in population:
        cache_file = output_dir / f"{ds}_cache.npz"
        if ds in checkpoints:
            chk = checkpoints[ds]
            contract_matches = checkpoint_contract_matches(
                chk, global_config_hash, db_sha256, run_population_sha
            )
            if chk.get("abstained") is True and contract_matches:
                reused_records.append({
                    "dataset": ds,
                    "file_path": None,
                    "file_hash": None,
                    "n_train": chk.get("n_train", 0),
                    "status": "abstained",
                })
                continue
            elif (
                cache_file.exists()
                and contract_matches
                and validate_cache(cache_file, global_config_hash, expected_dataset=ds)[0]
            ):
                actual_hash = get_sha256(cache_file)
                if actual_hash == chk["file_hash"]:
                    reused_records.append({
                        "dataset": ds,
                        "file_path": os.path.relpath(cache_file, PROJECT_DIR),
                        "file_hash": actual_hash,
                        "n_train": chk.get("n_train"),
                        "status": "reused",
                    })
                    continue
        to_process.append(ds)

    print(f"Reused {len(reused_records)} datasets from checkpoint. Processing {len(to_process)} datasets.")

    results = []
    errors = []

    if to_process:
        if args.workers <= 1:
            with checkpoint_file.open("a", buffering=1) as chk_f:
                for ds in to_process:
                    res = process_single_dataset(ds, output_dir, global_config_hash)
                    if res["success"]:
                        results.append(res)
                        chk_row = {
                            "dataset": res["dataset"],
                            "file_path": res["file_path"],
                            "config_hash": global_config_hash,
                            "database_sha256": db_sha256,
                            "population_sha256": run_population_sha,
                            "file_hash": res["file_hash"],
                            "abstained": res["abstained"],
                            "n_train": res["n_train"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        chk_f.write(json.dumps(chk_row) + "\n")
                        print(f"[SUCCESS] {ds} (abstained={res['abstained']})")
                    else:
                        errors.append(res)
                        print(f"[ERROR] {ds} failed: {res['error']}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_single_dataset, ds, output_dir, global_config_hash): ds
                    for ds in to_process
                }
                with checkpoint_file.open("a", buffering=1) as chk_f:
                    for future in as_completed(futures):
                        ds_name = futures[future]
                        try:
                            res = future.result()
                            if res["success"]:
                                results.append(res)
                                chk_row = {
                                    "dataset": res["dataset"],
                                    "file_path": res["file_path"],
                                    "config_hash": global_config_hash,
                                    "database_sha256": db_sha256,
                                    "population_sha256": run_population_sha,
                                    "file_hash": res["file_hash"],
                                    "abstained": res["abstained"],
                                    "n_train": res["n_train"],
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                chk_f.write(json.dumps(chk_row) + "\n")
                                print(f"[SUCCESS] {ds_name} (abstained={res['abstained']})")
                            else:
                                errors.append(res)
                                print(f"[ERROR] {ds_name} failed: {res['error']}")
                        except Exception as e:
                            tb = traceback.format_exc()
                            err_row = {
                                "dataset": ds_name,
                                "success": False,
                                "abstained": False,
                                "file_path": None,
                                "file_hash": None,
                                "n_train": 0,
                                "error": str(e),
                                "traceback": tb,
                            }
                            errors.append(err_row)
                            print(f"[CRITICAL ERROR] {ds_name} failed: {e}")

    # Generate Audit Artifacts
    print("Generating audit artifacts...")

    # Combine reused and computed
    all_manifest = []
    for r in reused_records:
        all_manifest.append(r)
    for r in results:
        all_manifest.append({
            "dataset": r["dataset"],
            "file_path": r["file_path"],
            "file_hash": r["file_hash"],
            "n_train": r["n_train"],
            "status": "abstained" if r["abstained"] else "computed",
        })
    all_manifest.sort(key=lambda x: x["dataset"])

    # 01_audit_contract.json
    contract_data = {
        "experiment_id": "exp164_train_normal_agreement_risk_audit",
        "global_config_hash": global_config_hash,
        "exp162_manifest_sha256": EXP162_MANIFEST_EXPECTED_HASH,
        "exp162_contract_sha256": EXP162_CONTRACT_EXPECTED_HASH,
        "database_path": str(DB_PATH),
        "database_sha256": db_sha256,
        "population_size": len(population),
        "frozen_population_sha256": frozen_population_sha,
        "run_population_sha256": run_population_sha,
        "alphas": [0.005, 0.01, 0.02, 0.05],
        "variants": ["train_audit_2of3", "train_audit_3of3"],
        "risk_targets": [0.005, 0.01, 0.02],
        "wilson_confidence": 0.95,
        "contains_test_values": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    with (output_dir / "01_audit_contract.json").open("w") as f:
        json.dump(contract_data, f, indent=2)

    # 02_cache_manifest.csv
    with (output_dir / "02_cache_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "file_path", "file_hash", "n_train", "status"])
        writer.writeheader()
        writer.writerows(all_manifest)

    # 03_errors.csv
    with (output_dir / "03_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "error", "traceback"])
        writer.writeheader()
        for err in errors:
            writer.writerow({
                "dataset": err["dataset"],
                "error": err["error"],
                "traceback": err["traceback"]
            })

    manifest_names = [row["dataset"] for row in all_manifest]
    missing_datasets = sorted(set(population) - set(manifest_names))
    unexpected_datasets = sorted(set(manifest_names) - set(population))
    duplicate_datasets = sorted({name for name in manifest_names if manifest_names.count(name) > 1})
    if errors or missing_datasets or unexpected_datasets or duplicate_datasets:
        blocker = {
            "experiment_id": "exp164_train_normal_agreement_risk_audit",
            "errors": len(errors),
            "missing_datasets": missing_datasets,
            "unexpected_datasets": unexpected_datasets,
            "duplicate_datasets": duplicate_datasets,
        }
        with (output_dir / "BLOCKER_REPORT.json").open("w") as f:
            json.dump(blocker, f, indent=2)
        raise SystemExit(1)

    # Read and aggregate caches for summaries
    from experiments.exp164.train_normal_risk_core import wilson_upper_bound, n_train_bin

    alphas = (0.005, 0.01, 0.02, 0.05)
    sources = ("spectral", "shapelet", "interval")
    variants = ("train_audit_2of3", "train_audit_3of3")

    dataset_alpha_summary = []
    global_runs = 0
    # keys: (alpha, source_or_variant) -> [false_reviews, total_runs]
    global_counts = {}

    # Store lists of flags to compute Jaccard and Phi
    # keys: alpha -> source -> list of boolean arrays
    all_flags = {a: {src: [] for src in sources} for a in alphas}
    threshold_resolution_rows = []

    for row in all_manifest:
        if row["status"] == "abstained":
            continue
        filepath = PROJECT_DIR / row["file_path"]
        with np.load(filepath, allow_pickle=False) as data:
            n_train = int(row["n_train"])
            global_runs += n_train

            # shape: (n_train, 4)
            flags_src = {
                "spectral": data["flags_spectral"],
                "shapelet": data["flags_shapelet"],
                "interval": data["flags_interval"],
                "train_audit_2of3": data["agreement_2of3"],
                "train_audit_3of3": data["agreement_3of3"],
            }


            for src in sources:
                thresholds = data[f"thresholds_{src}"]
                unique_per_run = np.array(
                    [len(np.unique(row_values)) for row_values in thresholds],
                    dtype=np.int32,
                )
                threshold_resolution_rows.append({
                    "dataset": row["dataset"],
                    "n_train": n_train,
                    "n_train_bin": n_train_bin(n_train),
                    "source": src,
                    "unique_threshold_values": len(np.unique(thresholds)),
                    "mean_unique_alpha_thresholds_per_run": f"{float(np.mean(unique_per_run)):.6f}",
                    "pseudo_runs_all_alphas_same": int(np.sum(unique_per_run == 1)),
                    "pseudo_runs_total": n_train,
                    "all_alpha_same_fraction": f"{float(np.mean(unique_per_run == 1)):.6f}",
                })

            # Append flags for dependence calculations
            for idx, a in enumerate(alphas):
                for src in sources:
                    all_flags[a][src].append(flags_src[src][:, idx])

            # For each alpha, source/variant
            for idx, a in enumerate(alphas):
                # source summaries
                for src in list(sources) + list(variants):
                    flags = flags_src[src][:, idx]
                    false_revs = int(np.sum(flags))
                    rate = false_revs / n_train
                    w_upper = wilson_upper_bound(false_revs, n_train)

                    dataset_alpha_summary.append({
                        "dataset": row["dataset"],
                        "alpha": a,
                        "source_or_variant": src,
                        "false_reviews": false_revs,
                        "runs": n_train,
                        "rate": f"{rate:.6f}",
                        "wilson_upper_bound": f"{w_upper:.6f}",
                        "meets_0.005": w_upper <= 0.005,
                        "meets_0.010": w_upper <= 0.010,
                        "meets_0.020": w_upper <= 0.020,
                    })

                    # Accumulate global
                    key = (a, src)
                    if key not in global_counts:
                        global_counts[key] = [0, 0]
                    global_counts[key][0] += false_revs
                    global_counts[key][1] += n_train

    # Write 04_dataset_alpha_summary.csv
    if dataset_alpha_summary:
        with (output_dir / "04_dataset_alpha_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset_alpha_summary[0].keys())
            writer.writeheader()
            writer.writerows(dataset_alpha_summary)

    # 05_global_summary.csv
    global_rows = []
    for key, (false_revs, total_runs) in global_counts.items():
        a, src = key
        rate = false_revs / total_runs if total_runs > 0 else 0.0
        w_upper = wilson_upper_bound(false_revs, total_runs)

        # Calculate percent of datasets that met each target
        sub_rows = [x for x in dataset_alpha_summary if x["alpha"] == a and x["source_or_variant"] == src]
        total_eligible = len(sub_rows)
        meets_005 = sum(1 for x in sub_rows if x["meets_0.005"]) / total_eligible if total_eligible > 0 else 0.0
        meets_010 = sum(1 for x in sub_rows if x["meets_0.010"]) / total_eligible if total_eligible > 0 else 0.0
        meets_020 = sum(1 for x in sub_rows if x["meets_0.020"]) / total_eligible if total_eligible > 0 else 0.0

        global_rows.append({
            "alpha": a,
            "source_or_variant": src,
            "false_reviews": false_revs,
            "runs": total_runs,
            "rate": f"{rate:.6f}",
            "wilson_upper_bound": f"{w_upper:.6f}",
            "pct_datasets_meet_0.005": f"{meets_005*100:.2f}%",
            "pct_datasets_meet_0.010": f"{meets_010*100:.2f}%",
            "pct_datasets_meet_0.020": f"{meets_020*100:.2f}%",
        })
    global_rows.sort(key=lambda x: (x["alpha"], x["source_or_variant"]))
    if global_rows:
        with (output_dir / "05_global_summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=global_rows[0].keys())
            writer.writeheader()
            writer.writerows(global_rows)

    # 06_risk_target_sensitivity.csv
    sensitivity_rows = []
    total_population = len(population)
    abstentions = sum(1 for x in all_manifest if x["status"] == "abstained")
    eligible_count = total_population - abstentions

    for a in alphas:
        for var in variants:
            sub_rows = [x for x in dataset_alpha_summary if x["alpha"] == a and x["source_or_variant"] == var]
            n_eligible = len(sub_rows) # should match eligible_count

            for target in (0.005, 0.010, 0.020):
                field = f"meets_{target:.3f}"
                passed = sum(1 for x in sub_rows if x[field])
                passed_pct = passed / n_eligible if n_eligible > 0 else 0.0

                sensitivity_rows.append({
                    "alpha": a,
                    "variant": var,
                    "risk_target": target,
                    "eligible_datasets": n_eligible,
                    "abstained_datasets": abstentions,
                    "passed_datasets": passed,
                    "passed_percentage": f"{passed_pct*100:.2f}%",
                })
    if sensitivity_rows:
        with (output_dir / "06_risk_target_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sensitivity_rows[0].keys())
            writer.writeheader()
            writer.writerows(sensitivity_rows)

    # 07_source_dependence.csv
    dependence_rows = []
    for a in alphas:
        # Concatenate flags across all datasets
        concat_flags = {}
        for src in sources:
            if all_flags[a][src]:
                concat_flags[src] = np.concatenate(all_flags[a][src])
            else:
                concat_flags[src] = np.array([], dtype=bool)

        # Compute pairwise
        src_pairs = [("spectral", "shapelet"), ("spectral", "interval"), ("shapelet", "interval")]
        for s1, s2 in src_pairs:
            f1 = concat_flags[s1]
            f2 = concat_flags[s2]

            # calculate jaccard and phi
            tp = np.sum(f1 & f2)
            fp = np.sum(f1 & ~f2)
            fn = np.sum(~f1 & f2)
            tn = np.sum(~f1 & ~f2)
            union = tp + fp + fn
            jaccard = tp / union if union > 0 else 1.0

            denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            phi = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

            dependence_rows.append({
                "alpha": a,
                "source_1": s1,
                "source_2": s2,
                "phi_correlation": f"{phi:.6f}",
                "jaccard_similarity": f"{jaccard:.6f}",
            })
    if dependence_rows:
        with (output_dir / "07_source_dependence.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dependence_rows[0].keys())
            writer.writeheader()
            writer.writerows(dependence_rows)

    # 08_threshold_resolution.csv
    if threshold_resolution_rows:
        with (output_dir / "08_threshold_resolution.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=threshold_resolution_rows[0].keys())
            writer.writeheader()
            writer.writerows(threshold_resolution_rows)

    # 09_n_train_bin_summary.csv
    bin_summary_rows = []
    for alpha in alphas:
        for source_or_variant in list(sources) + list(variants):
            matching_rows = [
                item for item in dataset_alpha_summary
                if item["alpha"] == alpha and item["source_or_variant"] == source_or_variant
            ]
            for bin_name in ("5_19", "20_49", "50_99", "ge100"):
                bin_rows = [
                    item for item in matching_rows
                    if n_train_bin(int(item["runs"])) == bin_name
                ]
                false_reviews = sum(int(item["false_reviews"]) for item in bin_rows)
                runs = sum(int(item["runs"]) for item in bin_rows)
                bin_summary_rows.append({
                    "alpha": alpha,
                    "source_or_variant": source_or_variant,
                    "n_train_bin": bin_name,
                    "datasets": len(bin_rows),
                    "normal_runs": runs,
                    "false_reviews": false_reviews,
                    "false_reviews_per_run": f"{(false_reviews / runs) if runs else 0.0:.6f}",
                    "wilson_upper_bound": f"{wilson_upper_bound(false_reviews, runs):.6f}",
                })
    with (output_dir / "09_n_train_bin_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bin_summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(bin_summary_rows)

    # Manifest file inventory
    artifact_files = [
        "01_audit_contract.json",
        "02_cache_manifest.csv",
        "03_errors.csv",
        "04_dataset_alpha_summary.csv",
        "05_global_summary.csv",
        "06_risk_target_sensitivity.csv",
        "07_source_dependence.csv",
        "08_threshold_resolution.csv",
        "09_n_train_bin_summary.csv",
        "CHECKPOINT.jsonl"
    ]

    manifest_entries = []
    for f_name in artifact_files:
        f_path = output_dir / f_name
        if f_path.exists():
            manifest_entries.append({
                "file_path": os.path.relpath(f_path, PROJECT_DIR),
                "sha256": get_sha256(f_path),
                "bytes": f_path.stat().st_size
            })

    with (output_dir / "MANIFEST.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_entries)

    with (output_dir / "MANIFEST.md").open("w") as f:
        f.write("# Exp164 TRAIN-Normal Agreement Risk Audit Manifest\n\n")
        f.write("This manifest documents the results of the Exp164 calibration risk audit.\n\n")
        f.write("## Setup details\n")
        f.write(f"- **Global Config Hash:** `{global_config_hash}`\n")
        f.write(f"- **Database SHA256:** `{db_sha256}`\n")
        f.write(f"- **Total Population:** {total_population}\n")
        f.write(f"- **Total Success:** {len(all_manifest)} (Abstained: {abstentions})\n")
        f.write(f"- **Total Errors:** {len(errors)}\n\n")
        f.write("## File Inventory\n\n")
        f.write("| File Path | SHA256 | Size (Bytes) |\n")
        f.write("| --- | --- | --- |\n")
        for entry in manifest_entries:
            f.write(f"| `{entry['file_path']}` | `{entry['sha256']}` | {entry['bytes']} |\n")

    print("\n=== RUN COMPLETE ===")
    print(f"Total Expected: {len(population)}")
    print(f"Total Success: {len(all_manifest)} (Abstained: {abstentions})")
    print(f"Total Errors: {len(errors)}")
    if len(errors) > 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    run_experiment()
