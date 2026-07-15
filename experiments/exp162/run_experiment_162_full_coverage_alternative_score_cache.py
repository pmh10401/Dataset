#!/usr/bin/env python3
"""
Exp162 Checkpointed Runner for Full-Coverage Alternative Score Cache.
"""
from __future__ import annotations

import os
# Set single-thread environment variables before importing numpy
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
from experiments.exp162.full_coverage_score_core import source_config_hash
from experiments.exp162.cache_io import validate_cache

EXP160_PRED_PATH = PROJECT_DIR / "outputs" / "exp160_candidate_zero_similarity_recovery" / "02_exp160_predictions.jsonl"
EXP160_EXPECTED_HASH = "ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")

def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def load_frozen_population() -> list[str]:
    # Hash verification
    if not EXP160_PRED_PATH.exists():
        raise FileNotFoundError(f"Exp160 prediction file not found at {EXP160_PRED_PATH}")
    actual_hash = get_sha256(EXP160_PRED_PATH)
    if actual_hash != EXP160_EXPECTED_HASH:
        raise ValueError(
            f"Exp160 prediction file hash mismatch!\nExpected: {EXP160_EXPECTED_HASH}\nActual:   {actual_hash}"
        )

    datasets = set()
    with EXP160_PRED_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ds = row.get("dataset_name")
            if ds:
                datasets.add(ds)
    population = sorted(datasets)
    if len(population) != 1117:
        raise ValueError(f"Frozen population must contain 1117 datasets, got {len(population)}")
    return population

def population_list_sha256(population: list[str]) -> str:
    payload = ("\n".join(population) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def process_single_dataset(dataset_name: str, output_dir: Path, expected_config_hash: str) -> dict:
    """
    Process task for a single dataset.
    This function runs in a child process worker.
    """
    # Double check inner threads restriction inside worker
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    from experiments.exp162.exp162_bridge import build_dataset_cache
    from experiments.exp162.cache_io import write_cache_atomic, validate_cache
    import hashlib

    cache_file = output_dir / f"{dataset_name}_cache.npz"
    try:
        payload, metadata = build_dataset_cache(dataset_name)
        write_cache_atomic(cache_file, payload, metadata)

        # Verify validation immediately
        if not validate_cache(cache_file, expected_config_hash, expected_dataset=dataset_name):
            raise ValueError("Validation failed on written cache NPZ file")

        # File SHA256
        h = hashlib.sha256()
        with cache_file.open("rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        file_hash = h.hexdigest()

        # We record relative path to PROJECT_DIR safely using os.path.relpath
        rel_path = os.path.relpath(cache_file, PROJECT_DIR)

        return {
            "dataset": dataset_name,
            "success": True,
            "error": None,
            "traceback": None,
            "file_path": rel_path,
            "file_hash": file_hash,
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        rel_path = os.path.relpath(cache_file, PROJECT_DIR) if cache_file.exists() else None
        return {
            "dataset": dataset_name,
            "success": False,
            "error": str(e),
            "traceback": tb,
            "file_path": rel_path,
            "file_hash": None,
        }

def run_experiment():
    parser = argparse.ArgumentParser(description="Run Exp162 Alternative Score Cache Construction")
    parser.add_argument("--dataset-limit", type=int, default=None, help="Limit number of datasets to process")
    parser.add_argument("--workers", type=int, default=7, help="Number of parallel dataset workers")
    parser.add_argument("--reset-checkpoint", action="store_true", help="Delete existing checkpoint and cache files")
    parser.add_argument("--output-dir", type=str, default="outputs/exp162_full_coverage_alternative_score_cache", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_hash = source_config_hash()
    checkpoint_file = output_dir / "CHECKPOINT.jsonl"

    if args.reset_checkpoint:
        print("Resetting checkpoint and removing previous cache files...")
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        # Remove any existing npz cache files in output_dir
        for f in output_dir.glob("*_cache.npz"):
            f.unlink()

    full_population = load_frozen_population()
    population_file_sha256 = get_sha256(EXP160_PRED_PATH)
    population_names_sha256 = population_list_sha256(full_population)
    database_sha256 = get_sha256(DB_PATH)
    population = full_population
    print(f"Loaded frozen population: {len(population)} datasets.")

    if args.dataset_limit is not None:
        population = population[:args.dataset_limit]
        print(f"Limited population to first {len(population)} datasets.")

    # Load checkpoints
    checkpoints = {}
    reused_count = 0
    if checkpoint_file.exists():
        with checkpoint_file.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    checkpoints[row["dataset"]] = row
                except Exception:
                    continue

    # Filter population for datasets that need processing
    to_process = []
    reused_records = []
    for ds in population:
        cache_file = output_dir / f"{ds}_cache.npz"
        if ds in checkpoints:
            chk = checkpoints[ds]
            # Verify file exists, config hash matches, and file hash matches
            if (
                cache_file.exists()
                and chk.get("config_hash") == config_hash
                and chk.get("database_sha256") == database_sha256
                and chk.get("population_list_sha256") == population_names_sha256
                and validate_cache(cache_file, config_hash, expected_dataset=ds)
            ):
                actual_file_hash = get_sha256(cache_file)
                if actual_file_hash == chk["file_hash"]:
                    reused_count += 1
                    reused_records.append({
                        "dataset": ds,
                        "file_path": os.path.relpath(cache_file, PROJECT_DIR),
                        "file_hash": actual_file_hash,
                        "status": "reused"
                    })
                    continue
        to_process.append(ds)

    print(f"Resume summary: {reused_count} datasets reused from checkpoints. {len(to_process)} datasets to compute.")

    results = []
    errors = []

    # Process remaining datasets
    if to_process:
        if args.workers <= 1:
            print("Running in sequential (single worker) mode...")
            with checkpoint_file.open("a", buffering=1) as chk_f:
                for ds_name in to_process:
                    res = process_single_dataset(ds_name, output_dir, config_hash)
                    if res["success"]:
                        results.append(res)
                        chk_row = {
                            "dataset": res["dataset"],
                            "file_path": res["file_path"],
                            "config_hash": config_hash,
                            "database_sha256": database_sha256,
                            "population_list_sha256": population_names_sha256,
                            "file_hash": res["file_hash"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        chk_f.write(json.dumps(chk_row) + "\n")
                        print(f"[SUCCESS] {ds_name} cached.")
                    else:
                        errors.append(res)
                        print(f"[ERROR] {ds_name} failed: {res['error']}")
        else:
            print(f"Spawning {args.workers} workers...")
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_single_dataset, ds, output_dir, config_hash): ds
                    for ds in to_process
                }

                # Open checkpoint file in append mode
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
                                    "config_hash": config_hash,
                                    "database_sha256": database_sha256,
                                    "population_list_sha256": population_names_sha256,
                                    "file_hash": res["file_hash"],
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                chk_f.write(json.dumps(chk_row) + "\n")
                                print(f"[SUCCESS] {ds_name} cached.")
                            else:
                                errors.append(res)
                                print(f"[ERROR] {ds_name} failed: {res['error']}")
                        except Exception as e:
                            tb = traceback.format_exc()
                            err_row = {
                                "dataset": ds_name,
                                "success": False,
                                "error": str(e),
                                "traceback": tb,
                                "file_path": None,
                                "file_hash": None,
                            }
                            errors.append(err_row)
                            print(f"[CRITICAL ERROR] {ds_name} failed with exception: {e}")

    # Generate Audit Artifacts
    print("Generating audit artifacts...")

    # 01_source_contract.json
    contract_data = {
        "experiment_id": "exp162_full_coverage_alternative_score_cache",
        "source_config_hash": config_hash,
        "population_file_sha256": population_file_sha256,
        "population_list_sha256": population_names_sha256,
        "database_path": str(DB_PATH),
        "database_sha256": database_sha256,
        "population_size": len(population),
        "full_population_size": len(full_population),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "source_configurations": {
            "spectral_derivative": {
                "score_rule": "mean absolute robust deviation",
                "features": "raw+diff+diff2 mean/std/median/p90-p10/max-abs + 8 fft bands + centroid/entropy/peak"
            },
            "shapelet_normal_prototype": {
                "score_rule": "mean absolute robust deviation of 32 min-distance features",
                "prototypes": "32 train-only, lengths [8, 16, 32, 64], capped at 512, z-normalized"
            },
            "interval_quantile": {
                "score_rule": "mean absolute robust deviation",
                "intervals": 16,
                "views": "raw+diff+periodogram",
                "interval_features": "mean, std, min, max, q05, q25, q50, q75, q95, slope"
            }
        }
    }
    with (output_dir / "01_source_contract.json").open("w") as f:
        json.dump(contract_data, f, indent=2)

    # 02_cache_manifest.csv
    manifest_rows = []
    # 1. Add reused records
    for r in reused_records:
        manifest_rows.append({
            "dataset": r["dataset"],
            "file_path": r["file_path"],
            "file_hash": r["file_hash"],
            "status": "reused"
        })
    # 2. Add newly computed records
    for r in results:
        manifest_rows.append({
            "dataset": r["dataset"],
            "file_path": r["file_path"],
            "file_hash": r["file_hash"],
            "status": "computed"
        })
    # Sort manifest rows by dataset name
    manifest_rows.sort(key=lambda x: x["dataset"])

    with (output_dir / "02_cache_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "file_path", "file_hash", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)

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

    # 04_coverage_summary.csv
    total_expected = len(population)
    total_success = len(manifest_rows)
    total_errors = len(errors)
    coverage_pct = (total_success / total_expected * 100.0) if total_expected > 0 else 0.0

    summary_data = [{
        "total_expected": total_expected,
        "total_success": total_success,
        "total_reused": reused_count,
        "total_computed": len(results),
        "total_errors": total_errors,
        "coverage_percentage": f"{coverage_pct:.2f}%",
        "source_config_hash": config_hash,
        "database_sha256": database_sha256,
        "population_list_sha256": population_names_sha256,
    }]
    with (output_dir / "04_coverage_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)

    # Manifest file lists for MANIFEST.csv and MANIFEST.md
    artifact_files = [
        "01_source_contract.json",
        "02_cache_manifest.csv",
        "03_errors.csv",
        "04_coverage_summary.csv",
        "CHECKPOINT.jsonl"
    ]
    if (output_dir / "VERIFICATION_REPORT.md").exists():
        artifact_files.append("VERIFICATION_REPORT.md")
    manifest_entries = []
    for f_name in artifact_files:
        f_path = output_dir / f_name
        if f_path.exists():
            manifest_entries.append({
                "file_path": os.path.relpath(f_path, PROJECT_DIR),
                "sha256": get_sha256(f_path),
                "bytes": f_path.stat().st_size
            })

    # Add MANIFEST.csv
    with (output_dir / "MANIFEST.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(manifest_entries)

    # Add MANIFEST.md
    with (output_dir / "MANIFEST.md").open("w") as f:
        f.write("# Exp162 Alternative Score Cache Manifest\n\n")
        f.write("This manifest documents the retrospective score caches built for three alternative sources over the frozen 1,117-dataset population.\n\n")
        f.write("## Execution Environment & Setup\n")
        f.write(f"- **Source Config Hash:** `{config_hash}`\n")
        f.write(f"- **Frozen Population Hash:** `{EXP160_EXPECTED_HASH}`\n")
        f.write(f"- **Frozen Population List SHA256:** `{population_names_sha256}`\n")
        f.write(f"- **Database SHA256:** `{database_sha256}`\n")
        f.write(f"- **Total Datasets Expected:** {total_expected}\n")
        f.write(f"- **Total Successfully Cached:** {total_success} (Reused: {reused_count}, Computed: {len(results)})\n")
        f.write(f"- **Total Errors:** {total_errors}\n\n")
        f.write("## File Inventory\n\n")
        f.write("| File Path | SHA256 | Size (Bytes) |\n")
        f.write("| --- | --- | --- |\n")
        for entry in manifest_entries:
            f.write(f"| `{entry['file_path']}` | `{entry['sha256']}` | {entry['bytes']} |\n")

    # Final report
    print("\n=== RUN COMPLETE ===")
    print(f"Total Expected: {total_expected}")
    print(f"Total Success: {total_success} (Reused: {reused_count}, Computed: {len(results)})")
    print(f"Total Errors: {total_errors}")
    print(f"Coverage: {coverage_pct:.2f}%")
    print(f"Config Hash: {config_hash}")

    if total_errors > 0 or total_success < total_expected:
        print("[FAILURE] Incomplete coverage or errors encountered during execution.")
        sys.exit(1)
    else:
        print("[SUCCESS] All targets processed successfully.")
        sys.exit(0)

if __name__ == "__main__":
    run_experiment()
