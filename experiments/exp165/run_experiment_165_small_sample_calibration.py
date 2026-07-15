#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_progress import ExperimentProgress
from experiments.exp164.train_normal_risk_core import n_train_bin, wilson_upper_bound
from experiments.exp165.similarity_calibration_core import (
    SOURCES,
    agreement_flags,
    build_descriptor,
    descriptor_scaler,
    leave_one_out_location_scale,
    robust_normalize,
    scale_descriptor,
    select_donors,
    similarity_pooled_flags,
)


EXPERIMENT_ID = "experiment_165_small_sample_calibration"
LABEL = "Exp165 · small-sample TRAIN-only similarity calibration audit"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DB_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
EXP164_DIR = PROJECT_DIR / "outputs/exp164_train_normal_agreement_risk_audit"
EXP164_CONTRACT = EXP164_DIR / "01_audit_contract.json"
EXP164_CONTRACT_SHA256 = "cd5af2984efb88d85d8de380dacb5c11016d2e3f7c9a2dbaf61d9b98c78448e0"
EXP164_MANIFEST = EXP164_DIR / "02_cache_manifest.csv"
EXP164_MANIFEST_SHA256 = "a3226c4c30f5217440126ebab1b502fd11e3dceec64451b7be88437e845e439e"
ALPHAS = (0.005, 0.01, 0.02, 0.05)
MIN_POOL_REFERENCE = 200
TARGET_MIN_TRAIN = 5
TARGET_MAX_TRAIN_EXCLUSIVE = 20
DONOR_MIN_TRAIN = 20
SOURCE_KEYS = {source: f"query_score_{source}" for source in SOURCES}
LOCAL_VARIANT_KEYS = {"local_2of3": "agreement_2of3", "local_3of3": "agreement_3of3"}
CONFIG_VERSION = "exp165_similarity_pool_v1"


_DONOR_NORMALIZED: dict[str, dict[str, np.ndarray]] = {}
_DONOR_DESCRIPTORS: dict[str, np.ndarray] = {}
_DONOR_COUNTS: dict[str, dict[str, int]] = {}
_DESCRIPTOR_CENTER = np.empty(0)
_DESCRIPTOR_SCALE = np.empty(0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash() -> str:
    contract = {
        "version": CONFIG_VERSION,
        "alphas": ALPHAS,
        "minimum_pool_reference": MIN_POOL_REFERENCE,
        "target_train_range": [TARGET_MIN_TRAIN, TARGET_MAX_TRAIN_EXCLUSIVE],
        "donor_min_train": DONOR_MIN_TRAIN,
        "sources": SOURCES,
        "exp164_contract_sha256": EXP164_CONTRACT_SHA256,
        "exp164_manifest_sha256": EXP164_MANIFEST_SHA256,
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest()


def validate_frozen_inputs() -> list[dict[str, str]]:
    checks = (
        (DB_PATH, DB_SHA256),
        (EXP164_CONTRACT, EXP164_CONTRACT_SHA256),
        (EXP164_MANIFEST, EXP164_MANIFEST_SHA256),
    )
    for path, expected in checks:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {path}: {actual}")
    rows = list(csv.DictReader(EXP164_MANIFEST.open(newline="", encoding="utf-8")))
    if len(rows) != 1117:
        raise ValueError(f"expected 1117 manifest rows, got {len(rows)}")
    names = [row["dataset"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate dataset in Exp164 manifest")
    for row in rows:
        if row["status"] == "abstained":
            continue
        path = PROJECT_DIR / row["file_path"]
        if not path.exists() or sha256_file(path) != row["file_hash"]:
            raise ValueError(f"Exp164 cache integrity failure: {row['dataset']}")
    return sorted(rows, key=lambda row: row["dataset"])


def load_source_scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        scores = {source: np.asarray(data[key], dtype=np.float64) for source, key in SOURCE_KEYS.items()}
    lengths = {values.size for values in scores.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"invalid source score shapes: {path}")
    if not all(np.isfinite(values).all() for values in scores.values()):
        raise ValueError(f"non-finite source score: {path}")
    return scores


def build_donor_pack(donor_rows: list[dict[str, str]]):
    normalized = {}
    raw_descriptors = {}
    counts = {}
    for row in donor_rows:
        name = row["dataset"]
        scores = load_source_scores(PROJECT_DIR / row["file_path"])
        normalized[name] = {source: robust_normalize(scores[source]) for source in SOURCES}
        raw_descriptors[name] = build_descriptor(scores)
        counts[name] = {source: int(normalized[name][source].size) for source in SOURCES}
    center, scale = descriptor_scaler(raw_descriptors)
    descriptors = {
        name: scale_descriptor(descriptor, center, scale)
        for name, descriptor in raw_descriptors.items()
    }
    return normalized, descriptors, counts, center, scale


def initialize_worker(normalized, descriptors, counts, center, scale):
    global _DONOR_NORMALIZED, _DONOR_DESCRIPTORS, _DONOR_COUNTS, _DESCRIPTOR_CENTER, _DESCRIPTOR_SCALE
    _DONOR_NORMALIZED = normalized
    _DONOR_DESCRIPTORS = descriptors
    _DONOR_COUNTS = counts
    _DESCRIPTOR_CENTER = center
    _DESCRIPTOR_SCALE = scale


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> str:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)
    return sha256_file(path)


def valid_existing_target_cache(path: Path, row: dict[str, str], expected_config_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            return bool(
                metadata.get("experiment_id") == EXPERIMENT_ID
                and metadata.get("dataset") == row["dataset"]
                and int(metadata.get("n_train")) == int(row["n_train"])
                and metadata.get("config_hash") == expected_config_hash
                and metadata.get("source_cache_sha256") == row["file_hash"]
                and metadata.get("contains_test_values") is False
                and metadata.get("contains_test_labels") is False
                and metadata.get("contains_evaluation_metrics") is False
            )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def process_target(row: dict[str, str], cache_dir: Path, expected_config_hash: str) -> dict:
    name = row["dataset"]
    try:
        source_path = PROJECT_DIR / row["file_path"]
        with np.load(source_path, allow_pickle=False) as data:
            scores = {source: np.asarray(data[SOURCE_KEYS[source]], dtype=np.float64) for source in SOURCES}
            local_2of3 = np.asarray(data["agreement_2of3"], dtype=bool)
            local_3of3 = np.asarray(data["agreement_3of3"], dtype=bool)
            local_thresholds = {
                source: np.asarray(data[f"thresholds_{source}"], dtype=np.float64) for source in SOURCES
            }
        n_train = int(row["n_train"])
        if any(values.shape != (n_train,) for values in scores.values()):
            raise ValueError("target score shape mismatch")
        if local_2of3.shape != (n_train, len(ALPHAS)) or local_3of3.shape != local_2of3.shape:
            raise ValueError("local agreement shape mismatch")

        pooled_thresholds = {source: np.zeros((n_train, len(ALPHAS))) for source in SOURCES}
        pooled_flags = {source: np.zeros((n_train, len(ALPHAS)), dtype=bool) for source in SOURCES}
        pooled_2of3 = np.zeros((n_train, len(ALPHAS)), dtype=bool)
        pooled_3of3 = np.zeros((n_train, len(ALPHAS)), dtype=bool)
        selected_donors = []
        selected_counts = np.zeros(n_train, dtype=np.int32)
        reference_counts = {source: np.zeros(n_train, dtype=np.int32) for source in SOURCES}

        for query_index in range(n_train):
            fit_scores = {source: np.delete(scores[source], query_index) for source in SOURCES}
            descriptor = scale_descriptor(
                build_descriptor(fit_scores), _DESCRIPTOR_CENTER, _DESCRIPTOR_SCALE
            )
            donors = select_donors(
                descriptor,
                _DONOR_DESCRIPTORS,
                _DONOR_COUNTS,
                minimum_reference=MIN_POOL_REFERENCE,
            )
            references = {
                source: np.concatenate([_DONOR_NORMALIZED[donor][source] for donor in donors])
                for source in SOURCES
            }
            query_scores = {}
            for source in SOURCES:
                location, scale = leave_one_out_location_scale(scores[source], query_index)
                query_scores[source] = float((scores[source][query_index] - location) / scale)
                reference_counts[source][query_index] = references[source].size
                if references[source].size < MIN_POOL_REFERENCE:
                    raise ValueError("minimum pooled reference is unavailable")
            thresholds, flags = similarity_pooled_flags(query_scores, references, ALPHAS)
            for alpha_index, alpha in enumerate(ALPHAS):
                for source in SOURCES:
                    pooled_thresholds[source][query_index, alpha_index] = thresholds[alpha][source]
                    pooled_flags[source][query_index, alpha_index] = flags[alpha][source]
                pooled_2of3[query_index, alpha_index] = agreement_flags(flags[alpha], 2)
                pooled_3of3[query_index, alpha_index] = agreement_flags(flags[alpha], 3)
            selected_donors.append(";".join(donors))
            selected_counts[query_index] = len(donors)

        metadata = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": name,
            "n_train": n_train,
            "config_hash": expected_config_hash,
            "source_cache_sha256": row["file_hash"],
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        payload = {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "selected_donors": np.asarray(selected_donors),
            "selected_donor_counts": selected_counts,
            "local_2of3": local_2of3,
            "local_3of3": local_3of3,
            "pooled_2of3": pooled_2of3,
            "pooled_3of3": pooled_3of3,
        }
        for source in SOURCES:
            payload[f"local_thresholds_{source}"] = local_thresholds[source]
            payload[f"pooled_thresholds_{source}"] = pooled_thresholds[source]
            payload[f"pooled_flags_{source}"] = pooled_flags[source]
            payload[f"reference_counts_{source}"] = reference_counts[source]
        cache_path = cache_dir / f"{name}.npz"
        output_hash = write_npz_atomic(cache_path, payload)
        return {
            "dataset": name,
            "n_train": n_train,
            "source_cache_sha256": row["file_hash"],
            "cache_path": str(cache_path.relative_to(PROJECT_DIR)),
            "cache_sha256": output_hash,
            "success": True,
            "error": None,
            "traceback": None,
        }
    except Exception as exc:
        return {
            "dataset": name,
            "n_train": int(row.get("n_train") or 0),
            "source_cache_sha256": row.get("file_hash"),
            "cache_path": None,
            "cache_sha256": None,
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def aggregate(output_dir: Path, target_rows: list[dict[str, str]], run_config_hash: str, errors: list[dict]):
    pool_rows = []
    dataset_rows = []
    resolution_rows = []
    transition_counts = {
        (alpha, variant): {"both": 0, "local_only": 0, "pooled_only": 0, "neither": 0}
        for alpha in ALPHAS
        for variant in ("2of3", "3of3")
    }
    global_counts = {}
    for row in target_rows:
        path = output_dir / "target_caches" / f"{row['dataset']}.npz"
        with np.load(path, allow_pickle=False) as data:
            n_train = int(row["n_train"])
            selected_donors = data["selected_donors"].astype(str)
            for index in range(n_train):
                pool_rows.append(
                    {
                        "dataset": row["dataset"],
                        "pseudo_run_index": index,
                        "n_train": n_train,
                        "selected_donor_count": int(data["selected_donor_counts"][index]),
                        "selected_donors": selected_donors[index],
                        "spectral_reference_count": int(data["reference_counts_spectral"][index]),
                        "shapelet_reference_count": int(data["reference_counts_shapelet"][index]),
                        "interval_reference_count": int(data["reference_counts_interval"][index]),
                    }
                )
            methods = {
                "local_exp164": {
                    "2of3": data["local_2of3"],
                    "3of3": data["local_3of3"],
                },
                "similarity_pool": {
                    "2of3": data["pooled_2of3"],
                    "3of3": data["pooled_3of3"],
                },
            }
            for source in SOURCES:
                for method in ("local_exp164", "similarity_pool"):
                    thresholds = data[f"{'local' if method == 'local_exp164' else 'pooled'}_thresholds_{source}"]
                    unique_per_run = np.asarray([len(np.unique(values)) for values in thresholds])
                    resolution_rows.append(
                        {
                            "dataset": row["dataset"],
                            "n_train": n_train,
                            "method": method,
                            "source": source,
                            "mean_unique_alpha_thresholds_per_run": float(np.mean(unique_per_run)),
                            "all_alphas_same_runs": int(np.sum(unique_per_run == 1)),
                            "runs": n_train,
                            "all_alphas_same_fraction": float(np.mean(unique_per_run == 1)),
                        }
                    )
            for alpha_index, alpha in enumerate(ALPHAS):
                for method, variants in methods.items():
                    for variant, matrix in variants.items():
                        flags = np.asarray(matrix[:, alpha_index], dtype=bool)
                        false_reviews = int(np.sum(flags))
                        upper = wilson_upper_bound(false_reviews, n_train)
                        dataset_rows.append(
                            {
                                "dataset": row["dataset"],
                                "n_train": n_train,
                                "n_train_bin": n_train_bin(n_train),
                                "method": method,
                                "alpha": alpha,
                                "variant": variant,
                                "false_reviews": false_reviews,
                                "runs": n_train,
                                "rate": false_reviews / n_train,
                                "wilson_upper_bound": upper,
                                "meets_0.005": upper <= 0.005,
                                "meets_0.010": upper <= 0.010,
                                "meets_0.020": upper <= 0.020,
                            }
                        )
                        counts = global_counts.setdefault((method, alpha, variant), [0, 0])
                        counts[0] += false_reviews
                        counts[1] += n_train
                for variant in ("2of3", "3of3"):
                    local = np.asarray(methods["local_exp164"][variant][:, alpha_index], dtype=bool)
                    pooled = np.asarray(methods["similarity_pool"][variant][:, alpha_index], dtype=bool)
                    transition_counts[(alpha, variant)]["both"] += int(np.sum(local & pooled))
                    transition_counts[(alpha, variant)]["local_only"] += int(np.sum(local & ~pooled))
                    transition_counts[(alpha, variant)]["pooled_only"] += int(np.sum(~local & pooled))
                    transition_counts[(alpha, variant)]["neither"] += int(np.sum(~local & ~pooled))

    global_rows = []
    for (method, alpha, variant), (false_reviews, runs) in sorted(global_counts.items()):
        matching = [
            item for item in dataset_rows
            if item["method"] == method and item["alpha"] == alpha and item["variant"] == variant
        ]
        global_rows.append(
            {
                "method": method,
                "alpha": alpha,
                "variant": variant,
                "datasets": len(matching),
                "false_reviews": false_reviews,
                "runs": runs,
                "rate": false_reviews / runs if runs else 0.0,
                "wilson_upper_bound": wilson_upper_bound(false_reviews, runs),
                "datasets_meet_0.005": sum(bool(item["meets_0.005"]) for item in matching),
                "datasets_meet_0.010": sum(bool(item["meets_0.010"]) for item in matching),
                "datasets_meet_0.020": sum(bool(item["meets_0.020"]) for item in matching),
            }
        )
    transition_rows = [
        {"alpha": alpha, "variant": variant, **counts}
        for (alpha, variant), counts in sorted(transition_counts.items())
    ]

    contract = {
        "experiment_id": EXPERIMENT_ID,
        "config_hash": run_config_hash,
        "database_path": str(DB_PATH),
        "database_sha256": DB_SHA256,
        "exp164_contract_sha256": EXP164_CONTRACT_SHA256,
        "exp164_manifest_sha256": EXP164_MANIFEST_SHA256,
        "population": 1117,
        "target_datasets": len(target_rows),
        "target_rule": "5 <= n_train < 20",
        "donor_rule": "n_train >= 20",
        "minimum_pool_reference": MIN_POOL_REFERENCE,
        "alphas": ALPHAS,
        "contains_test_values": False,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective_train_normal_calibration_audit": True,
    }
    (output_dir / "01_contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    write_csv(output_dir / "02_pool_manifest.csv", pool_rows)
    write_csv(output_dir / "03_dataset_alpha_summary.csv", dataset_rows)
    write_csv(output_dir / "04_global_summary.csv", global_rows)
    write_csv(output_dir / "05_threshold_resolution.csv", resolution_rows)
    write_csv(output_dir / "06_local_pool_transition.csv", transition_rows)
    write_csv(
        output_dir / "07_errors.csv",
        [{"dataset": item["dataset"], "error": item["error"], "traceback": item["traceback"]} for item in errors],
        ["dataset", "error", "traceback"],
    )
    manifest_rows = []
    for path in sorted(output_dir.glob("0*")):
        if path.is_file():
            manifest_rows.append({"file": str(path.relative_to(PROJECT_DIR)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "MANIFEST.csv", manifest_rows)
    return global_rows, len(pool_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Exp165 small-sample similarity calibration audit")
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("RANK_EXPERIMENT_WORKERS", "7")))
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--output-dir", default="outputs/exp165_small_sample_calibration")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "target_caches"
    cache_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "CHECKPOINT.jsonl"
    stdout_log = output_dir / "full_run.log"
    progress = None
    done = 0
    errors: list[dict] = []
    try:
        rows = validate_frozen_inputs()
        targets_all = [row for row in rows if TARGET_MIN_TRAIN <= int(row["n_train"]) < TARGET_MAX_TRAIN_EXCLUSIVE]
        donors = [row for row in rows if int(row["n_train"]) >= DONOR_MIN_TRAIN]
        targets = targets_all[: args.dataset_limit] if args.dataset_limit is not None else targets_all
        if len(targets_all) != 316 or len(donors) != 639:
            raise ValueError(f"population contract mismatch: targets={len(targets_all)} donors={len(donors)}")
        if args.reset_checkpoint:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()

        run_config_hash = config_hash()
        progress = ExperimentProgress(
            experiment_id=EXPERIMENT_ID,
            label=LABEL,
            expected_datasets=len(targets),
            detail_csv=str(output_dir / "03_dataset_alpha_summary.csv"),
            stdout_log=str(stdout_log),
            scope="retrospective TRAIN-normal calibration sensitivity",
            autonomous_metric="not_evaluated",
            review_metric="normal_false_review_audit",
        )
        progress.start(phase="prepare_donors")
        donor_pack = build_donor_pack(donors)

        checkpoint = {}
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    checkpoint[item["dataset"]] = item
        completed = []
        pending = []
        for row in targets:
            item = checkpoint.get(row["dataset"])
            cache_path = cache_dir / f"{row['dataset']}.npz"
            checkpoint_valid = (
                item
                and item.get("config_hash") == run_config_hash
                and item.get("source_cache_sha256") == row["file_hash"]
                and cache_path.exists()
                and sha256_file(cache_path) == item.get("cache_sha256")
            )
            if checkpoint_valid or valid_existing_target_cache(cache_path, row, run_config_hash):
                completed.append(row)
            else:
                pending.append(row)
        done = len(completed)
        progress.update(done=done, current_dataset=None, rows=done * len(ALPHAS) * 2, errors=0, phase="calibrate")
        print(f"Exp165 target={len(targets)} donors={len(donors)} reused={done} workers={args.workers}", flush=True)

        initialize_worker(*donor_pack)
        with checkpoint_path.open("a", encoding="utf-8", buffering=1) as checkpoint_file:
            if args.workers <= 1:
                iterator = ((row, process_target(row, cache_dir, run_config_hash)) for row in pending)
                executor = None
            else:
                executor = ProcessPoolExecutor(
                    max_workers=args.workers,
                    initializer=initialize_worker,
                    initargs=donor_pack,
                )
                futures = {executor.submit(process_target, row, cache_dir, run_config_hash): row for row in pending}
                iterator = ((futures[future], future.result()) for future in as_completed(futures))
            try:
                for row, result in iterator:
                    if result["success"]:
                        completed.append(row)
                        checkpoint_file.write(
                            json.dumps(
                                {
                                    "dataset": result["dataset"],
                                    "config_hash": run_config_hash,
                                    "source_cache_sha256": result["source_cache_sha256"],
                                    "cache_sha256": result["cache_sha256"],
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    else:
                        errors.append(result)
                    done += 1
                    rows_done = done * len(ALPHAS) * 2
                    print(
                        f"Progress: [ {done}/{len(targets)} ] rows={rows_done} dataset={row['dataset']} errors={len(errors)}",
                        flush=True,
                    )
                    progress.update(
                        done=done,
                        current_dataset=row["dataset"],
                        rows=rows_done,
                        errors=len(errors),
                        phase="calibrate",
                    )
            finally:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=False)

        completed_names = {row["dataset"] for row in completed}
        missing = sorted({row["dataset"] for row in targets} - completed_names)
        if errors or missing:
            blocker = {"experiment_id": EXPERIMENT_ID, "errors": errors, "missing": missing}
            (output_dir / "BLOCKER_REPORT.json").write_text(json.dumps(blocker, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(f"Exp165 incomplete: errors={len(errors)} missing={len(missing)}")

        progress.update(done=done, current_dataset=None, rows=done * len(ALPHAS) * 2, errors=0, phase="aggregate")
        global_rows, pool_rows = aggregate(output_dir, sorted(targets, key=lambda row: row["dataset"]), run_config_hash, errors)
        result_summary = {
            "errors": 0,
            "target_datasets": len(targets),
            "donor_datasets": len(donors),
            "normal_pseudo_runs": pool_rows,
            "global_rows": len(global_rows),
            "contains_evaluation_metrics": False,
        }
        progress.finish(rows=len(global_rows), result_summary=result_summary)
        print(f"{EXPERIMENT_ID} finished targets={len(targets)} pseudo_runs={pool_rows} errors=0", flush=True)
        return 0
    except Exception as exc:
        if progress is not None:
            progress.fail(error=str(exc), done=done, rows=done * len(ALPHAS) * 2, errors=max(1, len(errors)))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
