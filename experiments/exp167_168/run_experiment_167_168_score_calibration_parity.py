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
from experiments.exp164.train_normal_risk_core import fold_plan
from experiments.exp167_168.parity_core import (
    ALPHAS,
    SOURCE_CONFIGS,
    SOURCES,
    assert_label_free_prediction,
    candidate_rows,
    evaluate_indices,
    fold_paired_p_values,
)


AUDIT_EXPERIMENT_ID = "experiment_167_score_calibration_parity_audit"
EVALUATION_EXPERIMENT_ID = "experiment_168_fold_matched_parity_evaluation"
AUDIT_LABEL = "Exp167 · fold-matched score-calibration parity audit"
EVALUATION_LABEL = "Exp168 · fold-matched parity review sensitivity"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/exp167_168_score_calibration_parity"
EXP164_DIR = PROJECT_DIR / "outputs/exp164_train_normal_agreement_risk_audit"
EXP164_CONTRACT = EXP164_DIR / "01_audit_contract.json"
EXP164_MANIFEST = EXP164_DIR / "02_cache_manifest.csv"
EXP166_PREDICTIONS = PROJECT_DIR / "outputs/exp166_small_sample_anomaly_coverage/02_frozen_predictions.jsonl"
EXP164_CONTRACT_SHA256 = "cd5af2984efb88d85d8de380dacb5c11016d2e3f7c9a2dbaf61d9b98c78448e0"
EXP164_MANIFEST_SHA256 = "a3226c4c30f5217440126ebab1b502fd11e3dceec64451b7be88437e845e439e"
DB_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
TARGET_MIN_TRAIN = 5
TARGET_MAX_TRAIN_EXCLUSIVE = 20
EXPECTED_DATASETS = 316
PREDICTION_ROWS_PER_DATASET = len(ALPHAS) * len(SOURCE_CONFIGS)
FOLD_SEED = 20260715
WORKERS = 7
CONFIG_VERSION = "exp167_168_score_calibration_parity_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def frozen_config() -> dict:
    return {
        "version": CONFIG_VERSION,
        "alphas": ALPHAS,
        "sources": SOURCES,
        "source_configs": SOURCE_CONFIGS,
        "fold_method": "leave_one_out",
        "fold_seed": FOLD_SEED,
        "p_value": "(1 + paired_oof_ge_test) / (n_train + 1)",
        "finite_sample_correction": 1,
        "target_train_range": [TARGET_MIN_TRAIN, TARGET_MAX_TRAIN_EXCLUSIVE],
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
    rows = list(csv.DictReader(EXP164_MANIFEST.open(newline="", encoding="utf-8")))
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


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.shape != left.shape:
        return 0.0
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def score_dataset_folds(
    dataset: str,
    train_raw: list[np.ndarray],
    test_raw: list[np.ndarray],
) -> dict[str, object]:
    n_train = len(train_raw)
    n_test = len(test_raw)
    if not TARGET_MIN_TRAIN <= n_train < TARGET_MAX_TRAIN_EXCLUSIVE:
        raise ValueError(f"target n_train outside frozen range: {n_train}")
    plan = fold_plan(n_train, FOLD_SEED)
    if len(plan) != n_train:
        raise ValueError("small-sample parity requires leave-one-out folds")

    heldout = {source: np.empty(n_train, dtype=np.float64) for source in SOURCES}
    test_by_fold = {
        source: np.empty((n_train, n_test), dtype=np.float64) for source in SOURCES
    }
    fit_medians = {source: np.empty(n_train, dtype=np.float64) for source in SOURCES}
    target_lengths = np.empty(n_train, dtype=np.int32)

    for fold_index, (fit_indices, query_indices) in enumerate(plan):
        if len(query_indices) != 1:
            raise ValueError("leave-one-out fold must have one held-out TRAIN run")
        fit_raw = [train_raw[int(index)] for index in fit_indices]
        heldout_raw = [train_raw[int(query_indices[0])]]
        fold_scores, target_length = score_fold(
            dataset,
            fit_raw,
            heldout_raw + test_raw,
        )
        target_lengths[fold_index] = int(target_length)
        for source in SOURCES:
            fit_scores, query_scores = fold_scores[source]
            if query_scores.shape != (n_test + 1,):
                raise ValueError(f"{source} fold query score shape mismatch")
            heldout[source][fold_index] = float(query_scores[0])
            test_by_fold[source][fold_index] = query_scores[1:]
            fit_medians[source][fold_index] = float(np.median(fit_scores))

    full_scores, full_target_length = score_fold(dataset, train_raw, test_raw)
    full_train = {source: full_scores[source][0].astype(np.float64) for source in SOURCES}
    full_test = {source: full_scores[source][1].astype(np.float64) for source in SOURCES}

    arrays = [target_lengths]
    for source in SOURCES:
        arrays.extend(
            [heldout[source], test_by_fold[source], fit_medians[source], full_train[source], full_test[source]]
        )
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("non-finite parity score")
    return {
        "heldout": heldout,
        "test_by_fold": test_by_fold,
        "fit_medians": fit_medians,
        "full_train": full_train,
        "full_test": full_test,
        "target_lengths": target_lengths,
        "full_target_length": int(full_target_length),
        "n_train": n_train,
        "n_test": n_test,
    }


def _cache_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / "cache" / f"{dataset}_parity_cache.npz"


def _write_cache(path: Path, dataset: str, scored: dict[str, object], config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "dataset": np.asarray(dataset),
        "config_hash": np.asarray(config_hash),
        "n_train": np.asarray(scored["n_train"]),
        "n_test": np.asarray(scored["n_test"]),
        "target_lengths": np.asarray(scored["target_lengths"]),
        "full_target_length": np.asarray(scored["full_target_length"]),
    }
    for source in SOURCES:
        payload[f"heldout_{source}"] = np.asarray(scored["heldout"][source])
        payload[f"test_by_fold_{source}"] = np.asarray(scored["test_by_fold"][source])
        payload[f"fit_medians_{source}"] = np.asarray(scored["fit_medians"][source])
        payload[f"full_train_{source}"] = np.asarray(scored["full_train"][source])
        payload[f"full_test_{source}"] = np.asarray(scored["full_test"][source])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def _audit_worker(row: dict[str, str], output_dir: Path, config_hash: str) -> dict:
    dataset = row["dataset"]
    try:
        train_raw = load_values_only(dataset, "TRAIN")
        test_raw = load_values_only(dataset, "TEST")
        scored = score_dataset_folds(dataset, train_raw, test_raw)
        cache_path = _cache_path(output_dir, dataset)
        _write_cache(cache_path, dataset, scored, config_hash)
        source_rows = []
        for source in SOURCES:
            heldout = scored["heldout"][source]
            full_train = scored["full_train"][source]
            fold_test_median = np.median(scored["test_by_fold"][source], axis=0)
            full_test = scored["full_test"][source]
            source_rows.append(
                {
                    "dataset": dataset,
                    "source": source,
                    "n_train": len(train_raw),
                    "n_test": len(test_raw),
                    "heldout_median": float(np.median(heldout)),
                    "heldout_iqr": float(np.quantile(heldout, 0.75) - np.quantile(heldout, 0.25)),
                    "full_train_median": float(np.median(full_train)),
                    "full_train_iqr": float(np.quantile(full_train, 0.75) - np.quantile(full_train, 0.25)),
                    "median_shift": float(np.median(heldout) - np.median(full_train)),
                    "fold_test_variability_median": float(np.median(np.std(scored["test_by_fold"][source], axis=0))),
                    "fold_vs_full_test_rank_correlation": _rank_correlation(fold_test_median, full_test),
                }
            )
        audit_row = {
            "dataset": dataset,
            "n_train": len(train_raw),
            "n_test": len(test_raw),
            "folds": len(train_raw),
            "minimum_p_value": 1.0 / (len(train_raw) + 1.0),
            "resolution_blocked_0_005": int(0.005 < 1.0 / (len(train_raw) + 1.0)),
            "resolution_blocked_0_01": int(0.01 < 1.0 / (len(train_raw) + 1.0)),
            "resolution_blocked_0_02": int(0.02 < 1.0 / (len(train_raw) + 1.0)),
            "resolution_blocked_0_05": int(0.05 < 1.0 / (len(train_raw) + 1.0)),
            "target_length_min": int(np.min(scored["target_lengths"])),
            "target_length_max": int(np.max(scored["target_lengths"])),
            "full_target_length": int(scored["full_target_length"]),
            "cache_file": str(cache_path.relative_to(PROJECT_DIR)),
            "cache_sha256": sha256_file(cache_path),
            "config_hash": config_hash,
        }
        return {"dataset": dataset, "audit": audit_row, "sources": source_rows, "error": None}
    except Exception as exc:
        return {
            "dataset": dataset,
            "audit": None,
            "sources": [],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_audit(
    output_dir: Path,
    workers: int,
    dataset_limit: int | None,
    reset: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = target_rows(dataset_limit)
    config_hash = stable_hash(frozen_config())
    audit_rows: list[dict] = []
    source_rows: list[dict] = []
    errors: list[dict] = []
    progress = ExperimentProgress(
        experiment_id=AUDIT_EXPERIMENT_ID,
        label=AUDIT_LABEL,
        expected_datasets=len(targets),
        detail_csv=str(output_dir / "02_parity_audit.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective label-free score-calibration parity audit",
        autonomous_metric="not_evaluated",
        review_metric="not_evaluated",
    )
    progress.start(phase="fold-matched label-free audit")
    if reset:
        for row in targets:
            _cache_path(output_dir, row["dataset"]).unlink(missing_ok=True)

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(_audit_worker, row, output_dir, config_hash): row["dataset"]
            for row in targets
        }
        for future in as_completed(futures):
            result = future.result()
            if result["error"]:
                errors.append(result)
            else:
                audit_rows.append(result["audit"])
                source_rows.extend(result["sources"])
            done = len(audit_rows)
            progress.update(
                done=done,
                current_dataset=result["dataset"],
                rows=len(source_rows),
                errors=len(errors),
                phase="fold-matched label-free audit",
            )
            if (done + len(errors)) % 10 == 0 or done == len(targets):
                print(f"Exp167 audit [{done}/{len(targets)}] errors={len(errors)}", flush=True)

    if errors:
        write_csv(output_dir / "audit_errors.csv", errors)
        progress.fail(error=f"{len(errors)} audit errors", done=len(audit_rows), rows=len(source_rows), errors=len(errors))
        raise SystemExit(f"{len(errors)} audit errors")
    audit_rows.sort(key=lambda row: row["dataset"])
    source_rows.sort(key=lambda row: (row["dataset"], row["source"]))
    if len(audit_rows) != len(targets):
        raise RuntimeError("audit coverage incomplete")
    write_csv(output_dir / "02_parity_audit.csv", audit_rows)
    write_csv(output_dir / "03_fold_score_summary.csv", source_rows)
    result = {
        "datasets": len(audit_rows),
        "source_rows": len(source_rows),
        "errors": 0,
        "config_hash": config_hash,
        "resolution_blocked": {
            str(alpha): sum(alpha < 1.0 / (int(row["n_train"]) + 1.0) for row in audit_rows)
            for alpha in ALPHAS
        },
    }
    progress.finish(rows=len(source_rows), result_summary=result)
    return result


def _load_cache(path: Path, expected_dataset: str, config_hash: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        dataset = str(data["dataset"].item())
        cached_hash = str(data["config_hash"].item())
        if dataset != expected_dataset or cached_hash != config_hash:
            raise ValueError("parity cache contract mismatch")
        n_train = int(data["n_train"].item())
        n_test = int(data["n_test"].item())
        heldout = {source: np.asarray(data[f"heldout_{source}"], dtype=np.float64) for source in SOURCES}
        test_by_fold = {
            source: np.asarray(data[f"test_by_fold_{source}"], dtype=np.float64)
            for source in SOURCES
        }
        full_test = {source: np.asarray(data[f"full_test_{source}"], dtype=np.float64) for source in SOURCES}
    return {
        "n_train": n_train,
        "n_test": n_test,
        "heldout": heldout,
        "test_by_fold": test_by_fold,
        "full_test": full_test,
    }


def predict_dataset(row: dict[str, str], output_dir: Path, config_hash: str) -> dict:
    dataset = row["dataset"]
    try:
        cache_path = _cache_path(output_dir, dataset)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        cached = _load_cache(cache_path, dataset, config_hash)
        p_values = {
            source: fold_paired_p_values(
                cached["heldout"][source],
                cached["test_by_fold"][source],
            )
            for source in SOURCES
        }
        rows = candidate_rows(
            dataset=dataset,
            n_train=int(cached["n_train"]),
            n_test=int(cached["n_test"]),
            p_values=p_values,
            config_hash=config_hash,
        )
        for prediction in rows:
            prediction["source_p_value_min"] = {
                source: float(np.min(p_values[source])) for source in SOURCES
            }
            prediction["source_p_value_median"] = {
                source: float(np.median(p_values[source])) for source in SOURCES
            }
            prediction["parity_cache_sha256"] = sha256_file(cache_path)
            assert_label_free_prediction(prediction)
        return {"dataset": dataset, "rows": rows, "error": None}
    except Exception as exc:
        return {
            "dataset": dataset,
            "rows": [],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _checkpoint_rows(path: Path, config_hash: str) -> dict[str, list[dict]]:
    completed: dict[str, list[dict]] = {}
    if not path.exists():
        return completed
    for item in read_jsonl(path):
        rows = item.get("rows") or []
        if (
            len(rows) == PREDICTION_ROWS_PER_DATASET
            and all(row.get("config_hash") == config_hash for row in rows)
        ):
            for row in rows:
                assert_label_free_prediction(row)
            completed[item["dataset"]] = rows
    return completed


def validate_prediction_grid(predictions: list[dict], expected_datasets: int) -> None:
    expected_keys = {(float(alpha), source_config) for alpha in ALPHAS for source_config in SOURCE_CONFIGS}
    by_dataset: dict[str, list[dict]] = {}
    for row in predictions:
        assert_label_free_prediction(row)
        by_dataset.setdefault(row["dataset"], []).append(row)
    if len(by_dataset) != int(expected_datasets):
        raise ValueError("prediction dataset coverage mismatch")
    for dataset, rows in by_dataset.items():
        if len(rows) != PREDICTION_ROWS_PER_DATASET:
            raise ValueError(f"prediction row count mismatch for {dataset}")
        keys = [(float(row["alpha"]), row["source_config"]) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError(f"prediction grid mismatch for {dataset}")
        if len({int(row["n_train"]) for row in rows}) != 1:
            raise ValueError(f"n_train mismatch within {dataset}")
        if len({int(row["n_test"]) for row in rows}) != 1:
            raise ValueError(f"n_test mismatch within {dataset}")


def run_prediction(
    output_dir: Path,
    workers: int,
    dataset_limit: int | None,
    reset: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = target_rows(dataset_limit)
    config_hash = stable_hash(frozen_config())
    checkpoint = output_dir / "prediction_checkpoint.jsonl"
    if reset:
        checkpoint.unlink(missing_ok=True)
    completed = _checkpoint_rows(checkpoint, config_hash)
    pending = [row for row in targets if row["dataset"] not in completed]
    errors: list[dict] = []
    progress = ExperimentProgress(
        experiment_id=EVALUATION_EXPERIMENT_ID,
        label=EVALUATION_LABEL,
        expected_datasets=len(targets),
        detail_csv=str(output_dir / "06_exp168_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective fold-matched research review sensitivity",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.start(phase="label-free parity prediction")

    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {
            executor.submit(predict_dataset, row, output_dir, config_hash): row["dataset"]
            for row in pending
        }
        for future in as_completed(futures):
            result = future.result()
            dataset = result["dataset"]
            if result["error"]:
                errors.append(result)
            else:
                completed[dataset] = result["rows"]
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"dataset": dataset, "rows": result["rows"]}, sort_keys=True) + "\n")
            done = len(completed)
            progress.update(
                done=done,
                current_dataset=dataset,
                rows=done * PREDICTION_ROWS_PER_DATASET,
                errors=len(errors),
                phase="label-free parity prediction",
            )
            if (done + len(errors)) % 10 == 0 or done == len(targets):
                print(f"Exp168 prediction [{done}/{len(targets)}] errors={len(errors)}", flush=True)

    if errors:
        write_csv(output_dir / "prediction_errors.csv", errors)
        progress.fail(error=f"{len(errors)} prediction errors", done=len(completed), rows=len(completed) * PREDICTION_ROWS_PER_DATASET, errors=len(errors))
        raise SystemExit(f"{len(errors)} prediction errors")
    expected = {row["dataset"] for row in targets}
    if set(completed) != expected:
        raise RuntimeError("prediction coverage incomplete")
    predictions = [row for dataset in sorted(completed) for row in completed[dataset]]
    predictions.sort(key=lambda row: (row["dataset"], row["alpha"], row["source_config"]))
    validate_prediction_grid(predictions, expected_datasets=len(targets))
    prediction_path = output_dir / "04_frozen_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    manifest = {
        "experiment_id": EVALUATION_EXPERIMENT_ID,
        "dataset_count": len(targets),
        "prediction_rows": len(predictions),
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_hash": config_hash,
        "workers": int(workers),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "autonomous_metric_scope": "not_evaluated",
        "review_metric_scope": "human_review_candidate_research_only",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "05_prediction_manifest.json", manifest)
    progress.update(done=len(targets), current_dataset=None, rows=len(predictions), errors=0, phase="prediction frozen; awaiting evaluation")
    return manifest


def verify_prediction_manifest(output_dir: Path) -> tuple[dict, Path]:
    manifest = json.loads((output_dir / "05_prediction_manifest.json").read_text(encoding="utf-8"))
    prediction_path = output_dir / manifest["prediction_file"]
    actual_hash = sha256_file(prediction_path)
    if actual_hash != manifest["prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch")
    return manifest, prediction_path


def _exp166_local_map() -> dict[tuple[str, float, str], set[int]]:
    if not EXP166_PREDICTIONS.exists():
        return {}
    rows = read_jsonl(EXP166_PREDICTIONS)
    return {
        (row["dataset"], float(row["alpha"]), row["agreement"]): set(row["candidate_indices"])
        for row in rows
        if row.get("method") == "local_crossfit_reference"
    }


def run_evaluation(output_dir: Path, dataset_limit: int | None = None) -> dict:
    manifest, prediction_path = verify_prediction_manifest(output_dir)
    predictions = read_jsonl(prediction_path)
    validate_prediction_grid(predictions, expected_datasets=int(manifest["dataset_count"]))
    expected_rows = int(manifest["dataset_count"]) * PREDICTION_ROWS_PER_DATASET
    if len(predictions) != expected_rows:
        raise ValueError("prediction row coverage mismatch")

    labels_cache: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []
    for prediction in predictions:
        dataset = prediction["dataset"]
        if dataset not in labels_cache:
            labels_cache[dataset] = load_test_labels(dataset)
        labels = labels_cache[dataset]
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch for {dataset}")
        metrics = evaluate_indices(prediction["candidate_indices"], labels)
        metric_rows.append(
            {
                "dataset": dataset,
                "alpha": prediction["alpha"],
                "source_config": prediction["source_config"],
                "n_train": prediction["n_train"],
                "n_test": prediction["n_test"],
                "predicted_count": prediction["candidate_count"],
                **metrics,
                "resolution_blocked": int(prediction["resolution_blocked"]),
                "autonomous_metric_scope": "not_evaluated",
                "review_metric_scope": "retrospective_research_review_candidate",
            }
        )
    metric_rows.sort(key=lambda row: (row["dataset"], float(row["alpha"]), row["source_config"]))
    write_csv(output_dir / "06_exp168_dataset_metrics.csv", metric_rows)

    summary_rows: list[dict] = []
    for alpha in ALPHAS:
        for source_config in SOURCE_CONFIGS:
            group = [
                row
                for row in metric_rows
                if float(row["alpha"]) == alpha and row["source_config"] == source_config
            ]
            tp = sum(int(row["tp"]) for row in group)
            fp = sum(int(row["fp"]) for row in group)
            summary_rows.append(
                {
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
                }
            )
    write_csv(output_dir / "07_exp168_global_summary.csv", summary_rows)

    baseline = _exp166_local_map()
    prediction_map = {
        (row["dataset"], float(row["alpha"]), row["source_config"]): set(row["candidate_indices"])
        for row in predictions
    }
    transitions: list[dict] = []
    for dataset in sorted(labels_cache):
        labels = labels_cache[dataset]
        truth = set(np.flatnonzero(labels == 1).astype(int).tolist())
        for alpha in ALPHAS:
            for source_config in ("2of3", "3of3"):
                old = baseline.get((dataset, alpha, source_config), set())
                new = prediction_map[(dataset, alpha, source_config)]
                added = new - old
                removed = old - new
                transitions.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha,
                        "source_config": source_config,
                        "old_count": len(old),
                        "new_count": len(new),
                        "added_count": len(added),
                        "removed_count": len(removed),
                        "added_tp": len(added & truth),
                        "added_fp": len(added - truth),
                        "removed_tp": len(removed & truth),
                        "removed_fp": len(removed - truth),
                        "exact_jaccard": len(old & new) / len(old | new) if old | new else 1.0,
                    }
                )
    write_csv(output_dir / "08_candidate_transition.csv", transitions)
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
        detail_csv=str(output_dir / "06_exp168_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective fold-matched research review sensitivity",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.finish(rows=len(metric_rows), result_summary=result)
    return result


def write_contract(output_dir: Path) -> None:
    contract = {
        "audit_experiment_id": AUDIT_EXPERIMENT_ID,
        "evaluation_experiment_id": EVALUATION_EXPERIMENT_ID,
        "config": frozen_config(),
        "config_hash": stable_hash(frozen_config()),
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
    parser.add_argument("--phase", choices=("audit", "predict", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_contract(args.output_dir)
    if args.phase in {"audit", "all"}:
        result = run_audit(args.output_dir, args.workers, args.dataset_limit, args.reset_checkpoint)
        print(f"{AUDIT_EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}", flush=True)
    if args.phase in {"predict", "all"}:
        run_prediction(args.output_dir, args.workers, args.dataset_limit, args.reset_checkpoint)
    if args.phase in {"evaluate", "all"}:
        result = run_evaluation(args.output_dir, args.dataset_limit)
        print(f"{EVALUATION_EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
