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
import math
import sqlite3
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
from experiments.exp164.crossfit_sources import score_fold
from experiments.exp164.exp164_train_bridge import sanitize_series
from experiments.exp169_170.oob_core import balanced_oob_plan, aggregate_oob_scores
from experiments.exp169_170.pool_core import (
    build_descriptors,
    build_pool_manifest,
    pooled_p_values,
    fixed_candidate_rows,
    assert_label_free_prediction,
    SOURCES,
    ALPHAS,
    SOURCE_CONFIGS,
    POLICIES,
    SOURCE_PAIRS,
)

AUDIT_EXPERIMENT_ID = "experiment_169_oob_pooled_calibration_audit"
EVALUATION_EXPERIMENT_ID = "experiment_170_oob_pooled_review_evaluation"
AUDIT_LABEL = "Exp169 · OOB score pooled calibration audit"
EVALUATION_LABEL = "Exp170 · OOB pooled review evaluation"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/exp169_170_oob_pooled_calibration"
EXP164_DIR = PROJECT_DIR / "outputs/exp164_train_normal_agreement_risk_audit"
EXP164_CONTRACT = EXP164_DIR / "01_audit_contract.json"
EXP164_MANIFEST = EXP164_DIR / "02_cache_manifest.csv"
EXP166_PREDICTIONS = PROJECT_DIR / "outputs/exp166_small_sample_anomaly_coverage/02_frozen_predictions.jsonl"
EXP168_PREDICTIONS = PROJECT_DIR / "outputs/exp167_168_score_calibration_parity/04_frozen_predictions.jsonl"
EXP164_CONTRACT_SHA256 = "cd5af2984efb88d85d8de380dacb5c11016d2e3f7c9a2dbaf61d9b98c78448e0"
EXP164_MANIFEST_SHA256 = "a3226c4c30f5217440126ebab1b502fd11e3dceec64451b7be88437e845e439e"
DB_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
TARGET_MIN_TRAIN = 5
TARGET_MAX_TRAIN_EXCLUSIVE = 20
EXPECTED_DATASETS = 316
PREDICTION_ROWS_PER_DATASET = len(POLICIES) * len(ALPHAS) * len(SOURCE_CONFIGS) # 3 * 4 * 5 = 60
OOB_SEED = 20260715
WORKERS = 7
CONFIG_VERSION = "exp169_170_oob_pooled_calibration_v1"
SCORE_CONFIG_VERSION = "exp169_170_oob_score_cache_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def verify_database_hash() -> str:
    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)
    actual = sha256_file(DB_PATH)
    if actual != DB_SHA256:
        raise ValueError(f"database hash mismatch: {actual} != {DB_SHA256}")
    return actual


def project_relative_path(path: Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate)
    return str(candidate.resolve().relative_to(PROJECT_DIR.resolve()))


def frozen_score_config() -> dict:
    return {
        "version": SCORE_CONFIG_VERSION,
        "sources": SOURCES,
        "oob_models": 32,
        "holdout_fraction": 0.35,
        "oob_seed": OOB_SEED,
        "normalization": "median_max_scaled_mad_iqr_std",
        "aggregation": "balanced_oob_median_equal_component_count",
    }


def score_config_hash() -> str:
    return stable_hash(frozen_score_config())


def frozen_config(minimum_reference: int = 200) -> dict:
    if int(minimum_reference) < 1:
        raise ValueError("minimum_reference must be positive")
    return {
        "version": CONFIG_VERSION,
        "alphas": ALPHAS,
        "sources": SOURCES,
        "source_configs": SOURCE_CONFIGS,
        "pool_policies": POLICIES,
        "oob_models": 32,
        "holdout_fraction": 0.35,
        "oob_seed": OOB_SEED,
        "score_config_hash": score_config_hash(),
        "minimum_reference": int(minimum_reference),
        "donor_contribution_cap": 20,
        "candidate_lane": "research_review_candidate",
        "top_n_budget": None,
        "family_routing": False,
        "dataset_name_routing": False,
        "test_length_routing": False,
        "test_metric_selection": False,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_values_only(dataset: str, split: str) -> list[np.ndarray]:
    if split not in {"TRAIN", "TEST"}:
        raise ValueError(split)
    connection = sqlite3.connect(str(DB_PATH))
    try:
        rows = connection.execute(
            """
            SELECT i.values_blob
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = ?
            ORDER BY i.instance_index
            """,
            (dataset, split),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"no {split} values for {dataset}")
    return [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in rows]


def load_test_labels(dataset: str) -> np.ndarray:
    connection = sqlite3.connect(str(DB_PATH))
    try:
        rows = connection.execute(
            """
            SELECT i.label
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset,),
        ).fetchall()
    finally:
        connection.close()
    labels = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    if labels.size == 0 or not set(labels.tolist()) <= {0, 1}:
        raise ValueError(f"invalid TEST labels for {dataset}")
    return labels


def target_rows(dataset_limit: int | None = None) -> list[dict[str, str]]:
    for path, expected in (
        (EXP164_CONTRACT, EXP164_CONTRACT_SHA256),
        (EXP164_MANIFEST, EXP164_MANIFEST_SHA256),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"input hash mismatch for {path}: {actual}")
    with EXP164_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1117:
        raise ValueError(f"expected 1117 manifest rows, got {len(rows)}")
    targets = [
        row
        for row in rows
        if row["status"] == "computed"
        and TARGET_MIN_TRAIN <= int(row["n_train"]) < TARGET_MAX_TRAIN_EXCLUSIVE
    ]
    targets.sort(key=lambda row: row["dataset"])
    if len(targets) != EXPECTED_DATASETS:
        raise ValueError(f"expected {EXPECTED_DATASETS} targets, got {len(targets)}")
    if dataset_limit is not None:
        targets = targets[: int(dataset_limit)]
    return targets


def build_dataset_cache(
    dataset: str,
    train_raw: list[np.ndarray],
    test_raw: list[np.ndarray],
    config_hash: str,
) -> dict:
    n_train = len(train_raw)
    n_test = len(test_raw)
    plan = balanced_oob_plan(n_train)

    train_components = {source: np.empty((32, n_train), dtype=np.float64) for source in SOURCES}
    test_components = {source: np.empty((32, n_test), dtype=np.float64) for source in SOURCES}
    target_lengths = np.empty(32, dtype=np.int32)

    for split in plan:
        m = split.model_index
        fit_raw = [train_raw[idx] for idx in split.fit_indices]
        heldout_raw = [train_raw[idx] for idx in split.holdout_indices]

        fold_scores, target_length = score_fold(
            dataset,
            fit_raw,
            heldout_raw + test_raw,
        )
        target_lengths[m] = int(target_length)

        for source in SOURCES:
            fit_scores, query_scores = fold_scores[source]
            n_held = len(heldout_raw)
            if query_scores.shape != (n_held + n_test,):
                raise ValueError(f"{source} query score shape mismatch in model {m}")

            train_components[source][m, list(split.fit_indices)] = fit_scores
            train_components[source][m, list(split.holdout_indices)] = query_scores[:n_held]
            test_components[source][m, :] = query_scores[n_held:]

    aggregated_train = {}
    aggregated_test = {}
    q_d_value = None
    for source in SOURCES:
        agg_tr, agg_te, q_d = aggregate_oob_scores(plan, train_components[source], test_components[source])
        aggregated_train[source] = agg_tr
        aggregated_test[source] = agg_te
        q_d_value = q_d

    return {
        "dataset": dataset,
        "config_hash": config_hash,
        "n_train": n_train,
        "n_test": n_test,
        "q_d": q_d_value,
        "target_lengths": target_lengths,
        "mean_target_length": float(np.mean(target_lengths)),
        "train_components": train_components,
        "test_components": test_components,
        "aggregated_train": aggregated_train,
        "aggregated_test": aggregated_test,
    }


def _cache_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "cache" / f"{dataset}_oob_cache.npz"


def _write_cache(path: Path, cached: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": np.asarray(cached["dataset"]),
        "config_hash": np.asarray(cached["config_hash"]),
        "n_train": np.asarray(cached["n_train"]),
        "n_test": np.asarray(cached["n_test"]),
        "q_d": np.asarray(cached["q_d"]),
        "target_lengths": np.asarray(cached["target_lengths"]),
        "mean_target_length": np.asarray(cached["mean_target_length"]),
    }
    for source in SOURCES:
        payload[f"train_components_{source}"] = cached["train_components"][source]
        payload[f"test_components_{source}"] = cached["test_components"][source]
        payload[f"aggregated_train_{source}"] = cached["aggregated_train"][source]
        payload[f"aggregated_test_{source}"] = cached["aggregated_test"][source]

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _load_cache(path: Path, expected_dataset: str, config_hash: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        dataset = str(data["dataset"].item())
        cached_hash = str(data["config_hash"].item())
        if dataset != expected_dataset or cached_hash != config_hash:
            raise ValueError("OOB cache contract mismatch")

        n_train = int(data["n_train"].item())
        n_test = int(data["n_test"].item())
        q_d = int(data["q_d"].item())
        target_lengths = np.asarray(data["target_lengths"], dtype=np.int32)
        mean_target_length = float(data["mean_target_length"].item())

        train_components = {}
        test_components = {}
        aggregated_train = {}
        aggregated_test = {}
        for source in SOURCES:
            train_components[source] = np.asarray(data[f"train_components_{source}"], dtype=np.float64)
            test_components[source] = np.asarray(data[f"test_components_{source}"], dtype=np.float64)
            aggregated_train[source] = np.asarray(data[f"aggregated_train_{source}"], dtype=np.float64)
            aggregated_test[source] = np.asarray(data[f"aggregated_test_{source}"], dtype=np.float64)

    if target_lengths.shape != (32,):
        raise ValueError(f"target length shape mismatch: {target_lengths.shape}")
    if q_d < 1 or q_d > 32:
        raise ValueError(f"invalid OOB component count: {q_d}")
    for source in SOURCES:
        expected_shapes = {
            "train_components": (32, n_train),
            "test_components": (32, n_test),
            "aggregated_train": (n_train,),
            "aggregated_test": (n_test,),
        }
        arrays = {
            "train_components": train_components[source],
            "test_components": test_components[source],
            "aggregated_train": aggregated_train[source],
            "aggregated_test": aggregated_test[source],
        }
        for name, values in arrays.items():
            if values.shape != expected_shapes[name]:
                raise ValueError(
                    f"{source} {name} shape mismatch: {values.shape} != {expected_shapes[name]}"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{source} {name} contains non-finite values")

    return {
        "dataset": dataset,
        "config_hash": config_hash,
        "n_train": n_train,
        "n_test": n_test,
        "q_d": q_d,
        "target_lengths": target_lengths,
        "mean_target_length": mean_target_length,
        "train_components": train_components,
        "test_components": test_components,
        "aggregated_train": aggregated_train,
        "aggregated_test": aggregated_test,
    }


def load_validated_cache_packs(
    output_dir: Path,
    expected_datasets: list[str],
    config_hash: str,
    include_test: bool,
) -> dict[str, dict]:
    manifest_path = output_dir / "02_oob_cache_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = sorted(str(dataset) for dataset in expected_datasets)
    actual = sorted(row.get("dataset", "") for row in rows)
    if len(actual) != len(set(actual)) or actual != expected:
        raise ValueError(
            f"cache manifest dataset coverage mismatch: {len(actual)} rows for {len(expected)} targets"
        )

    packs: dict[str, dict] = {}
    for row in rows:
        dataset = row["dataset"]
        if row.get("status") != "computed":
            raise ValueError(f"cache status mismatch for {dataset}")
        if row.get("config_hash") != config_hash or row.get("score_config_hash") != config_hash:
            raise ValueError(f"cache config mismatch for {dataset}")
        cache_path = Path(row["cache_file"])
        if not cache_path.is_absolute():
            cache_path = PROJECT_DIR / cache_path
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        actual_hash = sha256_file(cache_path)
        if actual_hash != row.get("cache_sha256"):
            raise ValueError(f"cache hash mismatch for {dataset}")

        cached = _load_cache(cache_path, dataset, config_hash)
        if cached["n_train"] != int(row["n_train"]):
            raise ValueError(f"cache n_train mismatch for {dataset}")
        if cached["n_test"] != int(row["n_test"]):
            raise ValueError(f"cache n_test mismatch for {dataset}")
        if cached["q_d"] != int(row["q_d"]):
            raise ValueError(f"cache q_d mismatch for {dataset}")

        pack = {
            "n_train": cached["n_train"],
            "target_length": cached["mean_target_length"],
            "oob_scores": cached["aggregated_train"],
        }
        if include_test:
            pack.update(
                {
                    "aggregated_test": cached["aggregated_test"],
                    "n_test": cached["n_test"],
                }
            )
        packs[dataset] = pack
    return packs


def _cache_worker(row: dict[str, str], output_dir: Path, config_hash: str) -> dict:
    dataset = row["dataset"]
    try:
        train_raw = load_values_only(dataset, "TRAIN")
        test_raw = load_values_only(dataset, "TEST")
        scored = build_dataset_cache(dataset, train_raw, test_raw, config_hash)
        cache_path = _cache_path(output_dir, dataset)
        _write_cache(cache_path, scored)

        manifest_row = {
            "dataset": dataset,
            "status": "computed",
            "n_train": len(train_raw),
            "n_test": len(test_raw),
            "q_d": scored["q_d"],
            "mean_target_length": scored["mean_target_length"],
            "cache_file": project_relative_path(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "config_hash": config_hash,
            "score_config_hash": config_hash,
        }
        return {"dataset": dataset, "manifest": manifest_row, "error": None}
    except Exception as exc:
        return {
            "dataset": dataset,
            "manifest": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_cache(
    output_dir: Path,
    workers: int,
    dataset_limit: int | None,
    reset: bool,
    minimum_reference: int = 200,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    verify_database_hash()
    targets = target_rows(dataset_limit)
    config_hash = score_config_hash()

    manifest_rows = []
    errors = []

    if reset:
        for row in targets:
            _cache_path(output_dir, row["dataset"]).unlink(missing_ok=True)

    progress = ExperimentProgress(
        experiment_id=AUDIT_EXPERIMENT_ID,
        label=AUDIT_LABEL,
        expected_datasets=len(targets),
        detail_csv=str(output_dir / "02_oob_cache_manifest.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective label-free OOB score calibration audit",
        autonomous_metric="not_evaluated",
        review_metric="not_evaluated",
    )
    progress.start(phase="OOB caching")

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(_cache_worker, row, output_dir, config_hash): row["dataset"]
            for row in targets
        }
        for future in as_completed(futures):
            result = future.result()
            if result["error"]:
                errors.append(result)
            else:
                manifest_rows.append(result["manifest"])
            done = len(manifest_rows)
            progress.update(
                done=done,
                current_dataset=result["dataset"],
                rows=done,
                errors=len(errors),
                phase="OOB caching",
            )
            if (done + len(errors)) % 10 == 0 or done == len(targets):
                print(f"OOB caching [{done}/{len(targets)}] errors={len(errors)}", flush=True)

    if errors:
        write_csv(output_dir / "cache_errors.csv", errors)
        progress.fail(error=f"{len(errors)} cache errors", done=len(manifest_rows), rows=len(manifest_rows), errors=len(errors))
        raise SystemExit(f"{len(errors)} cache errors")

    manifest_rows.sort(key=lambda row: row["dataset"])
    if len(manifest_rows) != len(targets):
        raise RuntimeError("cache coverage incomplete")

    write_csv(output_dir / "02_oob_cache_manifest.csv", manifest_rows)
    progress.finish(rows=len(manifest_rows), result_summary={"datasets": len(manifest_rows)})
    return {"datasets": len(manifest_rows), "config_hash": config_hash}


def wilson_upper_bound(x: int, n: int, confidence: float = 0.95) -> float:
    if n == 0:
        return 0.0
    p = x / n
    z = 1.96
    denominator = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return (center + spread) / denominator


def build_pool_manifest_rows(
    packs: dict[str, dict],
    minimum_reference: int,
    donor_cap: int,
    policy_config_hash: str,
) -> tuple[list[dict], dict[str, dict]]:
    scaled_descriptors = build_descriptors(packs)
    rows: list[dict] = []
    pools_by_dataset: dict[str, dict] = {}
    for dataset in sorted(packs):
        pools = build_pool_manifest(
            dataset,
            packs,
            minimum_reference=minimum_reference,
            donor_cap=donor_cap,
            scaled_descriptors=scaled_descriptors,
        )
        pools_by_dataset[dataset] = pools
        for policy, pool in pools.items():
            for source in SOURCES:
                reference = np.ascontiguousarray(pool.references[source], dtype=np.float64)
                rows.append(
                    {
                        "dataset": dataset,
                        "pool_policy": policy,
                        "source": source,
                        "reference_size": int(reference.size),
                        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
                        "minimum_p_value": 1.0 / (reference.size + 1.0),
                        "donor_count": len(pool.donors),
                        "donors": ",".join(pool.donors),
                        "per_donor_counts": json.dumps(
                            pool.per_donor_counts[source],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "minimum_reference": int(minimum_reference),
                        "donor_cap": int(donor_cap),
                        "pool_expansion_dc": int(pool.expansion_dc),
                        "pool_expansion_dq": int(pool.expansion_dq),
                        "policy_config_hash": policy_config_hash,
                    }
                )
    rows.sort(key=lambda row: (row["dataset"], row["pool_policy"], row["source"]))
    return rows, pools_by_dataset


def validate_pool_manifest_file(
    output_dir: Path,
    packs: dict[str, dict],
    minimum_reference: int,
    donor_cap: int,
    policy_config_hash: str,
) -> dict[str, dict]:
    path = output_dir / "03_pool_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    expected_rows, pools_by_dataset = build_pool_manifest_rows(
        packs,
        minimum_reference=minimum_reference,
        donor_cap=donor_cap,
        policy_config_hash=policy_config_hash,
    )
    actual_keys = [
        (row.get("dataset"), row.get("pool_policy"), row.get("source"))
        for row in actual_rows
    ]
    expected_keys = [
        (row["dataset"], row["pool_policy"], row["source"])
        for row in expected_rows
    ]
    if len(actual_keys) != len(set(actual_keys)) or actual_keys != expected_keys:
        raise ValueError("pool manifest coverage or key mismatch")
    for actual, expected in zip(actual_rows, expected_rows):
        for field, expected_value in expected.items():
            if actual.get(field) != str(expected_value):
                raise ValueError(
                    f"pool manifest mismatch for {expected['dataset']} "
                    f"{expected['pool_policy']} {expected['source']} field {field}"
                )
    return pools_by_dataset


def run_audit(
    output_dir: Path,
    dataset_limit: int | None,
    minimum_reference: int = 200,
) -> dict:
    config_hash = score_config_hash()
    expected_datasets = [row["dataset"] for row in target_rows(dataset_limit)]
    packs = load_validated_cache_packs(
        output_dir,
        expected_datasets=expected_datasets,
        config_hash=config_hash,
        include_test=False,
    )

    policy_config_hash = stable_hash(frozen_config(minimum_reference))
    pool_rows, pools_by_dataset = build_pool_manifest_rows(
        packs,
        minimum_reference=minimum_reference,
        donor_cap=20,
        policy_config_hash=policy_config_hash,
    )
    dataset_audit_rows = []

    progress = ExperimentProgress(
        experiment_id=AUDIT_EXPERIMENT_ID,
        label=AUDIT_LABEL,
        expected_datasets=len(packs),
        detail_csv=str(output_dir / "04_exp169_dataset_audit.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective label-free OOB score calibration audit",
        autonomous_metric="not_evaluated",
        review_metric="not_evaluated",
    )
    progress.start(phase="audit generation")

    done = 0
    for d in sorted(packs.keys()):
        pools = pools_by_dataset[d]

        cached_train_scores = packs[d]["oob_scores"]
        n_train = packs[d]["n_train"]

        for policy in POLICIES:
            pool = pools[policy]

            train_p = {
                source: pooled_p_values(cached_train_scores[source], pool.references[source])
                for source in SOURCES
            }

            for alpha in ALPHAS:
                flags = {source: train_p[source] <= alpha for source in SOURCES}
                source_count = np.vstack([flags[source] for source in SOURCES]).sum(axis=0)
                config_flags = {
                    name: flags[sources[0]] & flags[sources[1]]
                    for name, sources in SOURCE_PAIRS.items()
                }
                config_flags["2of3"] = source_count >= 2
                config_flags["3of3"] = source_count >= 3

                for source_config in SOURCE_CONFIGS:
                    indices = np.flatnonzero(config_flags[source_config]).astype(int).tolist()
                    normal_review_count = len(indices)
                    normal_review_rate = normal_review_count / n_train
                    w_upper = wilson_upper_bound(normal_review_count, n_train)

                    config_sources = SOURCE_PAIRS.get(source_config, SOURCES)
                    ref_sizes = [pool.references[src].size for src in config_sources]
                    ref_size_min = min(ref_sizes)
                    ref_size_max = max(ref_sizes)
                    ref_size_mean = float(np.mean(ref_sizes))

                    minimum_p_val = 1.0 / (ref_size_min + 1.0)
                    res_blocked = int(float(alpha) < minimum_p_val)

                    dataset_audit_rows.append({
                        "dataset": d,
                        "pool_policy": policy,
                        "alpha": float(alpha),
                        "source_config": source_config,
                        "n_train": n_train,
                        "normal_review_count": normal_review_count,
                        "normal_review_rate": normal_review_rate,
                        "wilson_upper_bound": w_upper,
                        "candidate_zero": int(normal_review_count == 0),
                        "resolution_blocked": res_blocked,
                        "donor_count": len(pool.donors),
                        "reference_size_min": ref_size_min,
                        "reference_size_max": ref_size_max,
                        "reference_size_mean": ref_size_mean,
                        "pool_expansion_dc": pool.expansion_dc if policy == "coarse_strata_pool" else 0,
                        "pool_expansion_dq": pool.expansion_dq if policy == "coarse_strata_pool" else 0,
                    })
        done += 1
        progress.update(
            done=done,
            current_dataset=d,
            rows=len(dataset_audit_rows),
            errors=0,
            phase="audit generation",
        )

    write_csv(output_dir / "03_pool_manifest.csv", pool_rows)
    write_csv(output_dir / "04_exp169_dataset_audit.csv", dataset_audit_rows)

    summary_rows = []
    for policy in POLICIES:
        for alpha in ALPHAS:
            for source_config in SOURCE_CONFIGS:
                group = [
                    row for row in dataset_audit_rows
                    if row["pool_policy"] == policy and float(row["alpha"]) == alpha and row["source_config"] == source_config
                ]
                total_normal_instances = sum(row["n_train"] for row in group)
                total_normal_review = sum(row["normal_review_count"] for row in group)
                summary_rows.append({
                    "pool_policy": policy,
                    "alpha": alpha,
                    "source_config": source_config,
                    "datasets": len(group),
                    "total_normal_instances": total_normal_instances,
                    "total_normal_review_count": total_normal_review,
                    "mean_normal_review_rate": float(np.mean([row["normal_review_rate"] for row in group])),
                    "candidate_zero_datasets": sum(row["candidate_zero"] for row in group),
                    "resolution_blocked_datasets": sum(row["resolution_blocked"] for row in group),
                    "mean_donor_count": float(np.mean([row["donor_count"] for row in group])),
                    "mean_reference_size": float(np.mean([row["reference_size_mean"] for row in group])),
                })
    write_csv(output_dir / "05_exp169_global_summary.csv", summary_rows)

    result = {"datasets": len(packs), "errors": 0}
    progress.finish(rows=len(dataset_audit_rows), result_summary=result)
    return result


def run_prediction(
    output_dir: Path,
    workers: int,
    dataset_limit: int | None,
    reset: bool,
    minimum_reference: int = 200,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_config_hash = score_config_hash()
    policy_config_hash = stable_hash(frozen_config(minimum_reference))

    cache_manifest_path = output_dir / "02_oob_cache_manifest.csv"
    if not cache_manifest_path.exists():
        raise FileNotFoundError(f"Cache manifest not found: {cache_manifest_path}")
    pool_manifest_path = output_dir / "03_pool_manifest.csv"
    if not pool_manifest_path.exists():
        raise FileNotFoundError(f"Pool manifest not found: {pool_manifest_path}")

    expected_datasets = [row["dataset"] for row in target_rows(dataset_limit)]
    packs = load_validated_cache_packs(
        output_dir,
        expected_datasets=expected_datasets,
        config_hash=cache_config_hash,
        include_test=True,
    )
    pools_by_dataset = validate_pool_manifest_file(
        output_dir,
        packs,
        minimum_reference=minimum_reference,
        donor_cap=20,
        policy_config_hash=policy_config_hash,
    )

    progress = ExperimentProgress(
        experiment_id=EVALUATION_EXPERIMENT_ID,
        label=EVALUATION_LABEL,
        expected_datasets=len(packs),
        detail_csv=str(output_dir / "08_exp170_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective OOB pooled research review sensitivity",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.start(phase="frozen prediction generation")

    predictions = []
    done = 0
    for d in sorted(packs.keys()):
        pools = pools_by_dataset[d]

        p_values = {}
        reference_counts = {}
        for policy, pool in pools.items():
            p_values[policy] = {
                source: pooled_p_values(packs[d]["aggregated_test"][source], pool.references[source])
                for source in SOURCES
            }
            reference_counts[policy] = {
                source: pool.references[source].size
                for source in SOURCES
            }

        rows = fixed_candidate_rows(
            dataset=d,
            n_test=packs[d]["n_test"],
            p_values=p_values,
            config_hash=policy_config_hash,
            n_train=packs[d]["n_train"],
            reference_counts=reference_counts,
        )
        for row in rows:
            assert_label_free_prediction(row)

        predictions.extend(rows)
        done += 1
        progress.update(
            done=done,
            current_dataset=d,
            rows=len(predictions),
            errors=0,
            phase="frozen prediction generation",
        )

    predictions.sort(key=lambda r: (r["dataset"], r["pool_policy"], float(r["alpha"]), r["source_config"]))
    validate_prediction_grid(predictions, expected_datasets=len(packs))

    prediction_path = output_dir / "06_frozen_predictions.jsonl"
    write_jsonl(prediction_path, predictions)

    manifest = {
        "experiment_id": EVALUATION_EXPERIMENT_ID,
        "dataset_count": len(packs),
        "prediction_rows": len(predictions),
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "cache_manifest_file": cache_manifest_path.name,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "pool_manifest_file": pool_manifest_path.name,
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
        "config_hash": policy_config_hash,
        "score_config_hash": cache_config_hash,
        "workers": int(workers),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "autonomous_metric_scope": "not_evaluated",
        "review_metric_scope": "human_review_candidate_research_only",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_json(output_dir / "07_prediction_manifest.json", manifest)
    progress.update(done=len(packs), current_dataset=None, rows=len(predictions), errors=0, phase="prediction frozen; awaiting evaluation")
    return manifest


def validate_prediction_grid(rows: list[dict], expected_datasets: int) -> None:
    expected_keys = {
        (policy, float(alpha), source_config)
        for policy in POLICIES
        for alpha in ALPHAS
        for source_config in SOURCE_CONFIGS
    }
    datasets = sorted({str(row.get("dataset", "")) for row in rows})
    if "" in datasets or len(datasets) != int(expected_datasets):
        raise ValueError(
            f"prediction grid dataset coverage mismatch: {len(datasets)} != {expected_datasets}"
        )
    if len(rows) != int(expected_datasets) * PREDICTION_ROWS_PER_DATASET:
        raise ValueError("prediction grid row count mismatch")

    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        keys = [
            (row["pool_policy"], float(row["alpha"]), row["source_config"])
            for row in dataset_rows
        ]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError(f"prediction grid variant mismatch for {dataset}")
        n_tests = {int(row["n_test"]) for row in dataset_rows}
        if len(n_tests) != 1:
            raise ValueError(f"prediction grid n_test mismatch for {dataset}")
        for row in dataset_rows:
            assert_label_free_prediction(row)


def verify_prediction_manifest(output_dir: Path) -> tuple[dict, Path]:
    manifest = json.loads((output_dir / "07_prediction_manifest.json").read_text(encoding="utf-8"))
    contract_path = output_dir / "01_contract.json"
    if not contract_path.is_file():
        raise ValueError("experiment contract file missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("config_hash") != manifest.get("config_hash"):
        raise ValueError("contract config hash mismatch")
    prediction_path = output_dir / manifest["prediction_file"]
    actual_hash = sha256_file(prediction_path)
    if actual_hash != manifest["prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch")
    dependencies = (
        ("cache manifest", "cache_manifest_file", "cache_manifest_sha256"),
        ("pool manifest", "pool_manifest_file", "pool_manifest_sha256"),
    )
    for label, file_key, hash_key in dependencies:
        dependency_path = output_dir / manifest.get(file_key, "")
        if not dependency_path.is_file():
            raise ValueError(f"{label} file missing")
        if sha256_file(dependency_path) != manifest[hash_key]:
            raise ValueError(f"{label} hash mismatch")
    return manifest, prediction_path


def run_evaluation(output_dir: Path, dataset_limit: int | None = None) -> dict:
    manifest, prediction_path = verify_prediction_manifest(output_dir)
    predictions = read_jsonl(prediction_path)
    validate_prediction_grid(predictions, expected_datasets=int(manifest["dataset_count"]))
    verify_database_hash()

    labels_cache: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []

    from experiments.exp167_168.parity_core import evaluate_indices

    for prediction in predictions:
        dataset = prediction["dataset"]
        if dataset not in labels_cache:
            labels_cache[dataset] = load_test_labels(dataset)
        labels = labels_cache[dataset]
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch for {dataset}")

        metrics = evaluate_indices(prediction["candidate_indices"], labels)
        metric_rows.append({
            "dataset": dataset,
            "pool_policy": prediction["pool_policy"],
            "alpha": prediction["alpha"],
            "source_config": prediction["source_config"],
            "n_train": prediction["n_train"],
            "n_test": prediction["n_test"],
            "predicted_count": prediction["candidate_count"],
            **metrics,
            "resolution_blocked": int(prediction["resolution_blocked"]),
            "autonomous_metric_scope": "not_evaluated",
            "review_metric_scope": "retrospective_research_review_candidate",
        })

    metric_rows.sort(key=lambda r: (r["dataset"], r["pool_policy"], float(r["alpha"]), r["source_config"]))
    write_csv(output_dir / "08_exp170_dataset_metrics.csv", metric_rows)

    summary_rows = []
    for policy in POLICIES:
        for alpha in ALPHAS:
            for source_config in SOURCE_CONFIGS:
                group = [
                    row for row in metric_rows
                    if row["pool_policy"] == policy and float(row["alpha"]) == alpha and row["source_config"] == source_config
                ]
                tp = sum(int(row["tp"]) for row in group)
                fp = sum(int(row["fp"]) for row in group)
                summary_rows.append({
                    "pool_policy": policy,
                    "alpha": alpha,
                    "source_config": source_config,
                    "datasets": len(group),
                    "review_candidates": sum(int(row["predicted_count"]) for row in group),
                    "tp": tp,
                    "fp": fp,
                    "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
                    "mean_recall": float(np.mean([float(row["recall"]) for row in group])),
                    "mean_f1": float(np.mean([float(row["f1"]) for row in group])),
                    "zero_f1_datasets": sum(float(row["f1"]) == 0.0 for row in group),
                    "candidate_zero_datasets": sum(int(row["predicted_count"]) == 0 for row in group),
                    "mean_candidates_per_dataset": float(np.mean([int(row["predicted_count"]) for row in group])),
                    "resolution_blocked_datasets": sum(int(row["resolution_blocked"]) for row in group),
                    "autonomous_metric_scope": "not_evaluated",
                    "review_metric_scope": "retrospective_research_review_candidate",
                    "selected_policy": 0,
                })
    write_csv(output_dir / "09_exp170_global_summary.csv", summary_rows)

    # Comparisons with Exp166/168
    exp166_local_map = {}
    exp166_pool_map = {}
    if EXP166_PREDICTIONS.exists():
        for row in read_jsonl(EXP166_PREDICTIONS):
            method = row.get("method")
            if method == "local_crossfit_reference":
                exp166_local_map[(row["dataset"], float(row["alpha"]), row["agreement"])] = set(row["candidate_indices"])
            elif method == "similarity_pool_reference":
                exp166_pool_map[(row["dataset"], float(row["alpha"]), row["agreement"])] = set(row["candidate_indices"])

    exp168_map = {}
    if EXP168_PREDICTIONS.exists():
        for row in read_jsonl(EXP168_PREDICTIONS):
            exp168_map[(row["dataset"], float(row["alpha"]), row["source_config"])] = set(row["candidate_indices"])

    transitions = []
    for prediction in predictions:
        dataset = prediction["dataset"]
        policy = prediction["pool_policy"]
        alpha = float(prediction["alpha"])
        source_config = prediction["source_config"]
        new_set = set(prediction["candidate_indices"])

        truth = set(np.flatnonzero(labels_cache[dataset] == 1).astype(int).tolist())

        row = {
            "dataset": dataset,
            "pool_policy": policy,
            "alpha": alpha,
            "source_config": source_config,
            "new_count": len(new_set),
        }

        key = (dataset, alpha, source_config)
        if key in exp168_map:
            old_set = exp168_map[key]
            added = new_set - old_set
            removed = old_set - new_set
            row.update({
                "exp168_old_count": len(old_set),
                "exp168_added_count": len(added),
                "exp168_removed_count": len(removed),
                "exp168_added_tp": len(added & truth),
                "exp168_added_fp": len(added - truth),
                "exp168_removed_tp": len(removed & truth),
                "exp168_removed_fp": len(removed - truth),
                "exp168_exact_jaccard": len(old_set & new_set) / len(old_set | new_set) if old_set | new_set else 1.0,
            })
        else:
            row.update({
                "exp168_old_count": 0, "exp168_added_count": 0, "exp168_removed_count": 0,
                "exp168_added_tp": 0, "exp168_added_fp": 0, "exp168_removed_tp": 0, "exp168_removed_fp": 0,
                "exp168_exact_jaccard": 0.0,
            })

        if key in exp166_local_map:
            old_set = exp166_local_map[key]
            row["exp166_local_jaccard"] = len(old_set & new_set) / len(old_set | new_set) if old_set | new_set else 1.0
        else:
            row["exp166_local_jaccard"] = 0.0

        if key in exp166_pool_map:
            old_set = exp166_pool_map[key]
            row["exp166_pool_jaccard"] = len(old_set & new_set) / len(old_set | new_set) if old_set | new_set else 1.0
        else:
            row["exp166_pool_jaccard"] = 0.0

        transitions.append(row)

    write_csv(output_dir / "10_candidate_transition.csv", transitions)

    completion = {
        "experiment_id": EVALUATION_EXPERIMENT_ID,
        "dataset_count": len(labels_cache),
        "prediction_rows": len(predictions),
        "prediction_sha256": manifest["prediction_sha256"],
        "retrospective": True,
        "autonomous_metric_scope": "not_evaluated",
        "selected_policy": False,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "11_completion.json", completion)

    result = {
        "datasets": len(labels_cache),
        "prediction_rows": len(predictions),
        "summary_rows": len(summary_rows),
        "prediction_sha256": manifest["prediction_sha256"],
        "retrospective": True,
        "autonomous_metric_scope": "not_evaluated",
        "selected_policy": False,
    }

    progress = ExperimentProgress(
        experiment_id=EVALUATION_EXPERIMENT_ID,
        label=EVALUATION_LABEL,
        expected_datasets=len(labels_cache),
        detail_csv=str(output_dir / "08_exp170_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective OOB pooled research review sensitivity",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.finish(rows=len(metric_rows), result_summary=result)
    return result


def write_contract(output_dir: Path, minimum_reference: int = 200) -> None:
    config = frozen_config(minimum_reference)
    contract = {
        "audit_experiment_id": AUDIT_EXPERIMENT_ID,
        "evaluation_experiment_id": EVALUATION_EXPERIMENT_ID,
        "config": config,
        "config_hash": stable_hash(config),
        "score_config": frozen_score_config(),
        "score_config_hash": score_config_hash(),
        "database_path": str(DB_PATH),
        "database_sha256_expected": DB_SHA256,
        "exp164_contract_sha256": EXP164_CONTRACT_SHA256,
        "exp164_manifest_sha256": EXP164_MANIFEST_SHA256,
        "expected_datasets": EXPECTED_DATASETS,
        "prediction_rows_per_dataset": PREDICTION_ROWS_PER_DATASET,
        "prediction_uses_test_values": True,
        "prediction_uses_test_labels": False,
        "test_metric_policy_selection": False,
        "candidate_lane": "research_review_candidate",
        "retrospective": True,
        "prospective_validated": False,
    }
    write_json(output_dir / "01_contract.json", contract)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cache", "audit", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-reference", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase != "evaluate":
        write_contract(args.output_dir, minimum_reference=args.minimum_reference)

    if args.phase in {"cache", "all"}:
        result = run_cache(
            args.output_dir,
            args.workers,
            args.dataset_limit,
            args.reset_checkpoint,
            minimum_reference=args.minimum_reference,
        )
        print(f"Cache phase finished: {json.dumps(result, sort_keys=True)}", flush=True)
    if args.phase in {"audit", "all"}:
        result = run_audit(args.output_dir, args.dataset_limit, minimum_reference=args.minimum_reference)
        print(f"{AUDIT_EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}", flush=True)
    if args.phase in {"predict", "all"}:
        run_prediction(args.output_dir, args.workers, args.dataset_limit, args.reset_checkpoint, minimum_reference=args.minimum_reference)
    if args.phase in {"evaluate", "all"}:
        result = run_evaluation(args.output_dir, args.dataset_limit)
        print(f"{EVALUATION_EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
