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
from experiments.exp165.run_experiment_165_small_sample_calibration import (
    EXP164_MANIFEST,
    MIN_POOL_REFERENCE,
    SOURCE_KEYS,
    build_donor_pack,
    validate_frozen_inputs,
)
from experiments.exp165.similarity_calibration_core import (
    SOURCES,
    build_descriptor,
    scale_descriptor,
    select_donors,
)
from experiments.exp166.exp166_core import (
    AGREEMENTS,
    ALPHAS,
    METHODS,
    assert_label_free_prediction,
    candidate_rows,
    evaluate_indices,
    normalize_query_scores,
    stable_config_hash,
)


EXPERIMENT_ID = "experiment_166_small_sample_anomaly_coverage"
LABEL = "Exp166 · small-sample similarity calibration anomaly coverage"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/exp166_small_sample_anomaly_coverage"
TARGET_MIN_TRAIN = 5
TARGET_MAX_TRAIN_EXCLUSIVE = 20
DONOR_MIN_TRAIN = 20
WORKERS = 7
CONFIG_VERSION = "exp166_small_sample_anomaly_coverage_v1"

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


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
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
    if not set(labels.tolist()) <= {0, 1}:
        raise ValueError(f"non-binary TEST labels for {dataset}")
    return labels


def frozen_config_hash() -> str:
    return stable_config_hash(
        {
            "version": CONFIG_VERSION,
            "alphas": ALPHAS,
            "agreements": AGREEMENTS,
            "methods": METHODS,
            "minimum_pool_reference": MIN_POOL_REFERENCE,
            "target_train_range": [TARGET_MIN_TRAIN, TARGET_MAX_TRAIN_EXCLUSIVE],
            "donor_min_train": DONOR_MIN_TRAIN,
            "normalization": "target_train_oof_median_mad",
            "candidate_lane": "research_review_candidate",
            "top_n_budget": None,
            "family_routing": False,
            "test_metric_selection": False,
        }
    )


def initialize_worker(normalized, descriptors, counts, center, scale):
    global _DONOR_NORMALIZED, _DONOR_DESCRIPTORS, _DONOR_COUNTS, _DESCRIPTOR_CENTER, _DESCRIPTOR_SCALE
    _DONOR_NORMALIZED = normalized
    _DONOR_DESCRIPTORS = descriptors
    _DONOR_COUNTS = counts
    _DESCRIPTOR_CENTER = center
    _DESCRIPTOR_SCALE = scale


def _target_oof_scores(row: dict[str, str]) -> dict[str, np.ndarray]:
    path = PROJECT_DIR / row["file_path"]
    with np.load(path, allow_pickle=False) as data:
        return {
            source: np.asarray(data[SOURCE_KEYS[source]], dtype=np.float64)
            for source in SOURCES
        }


def predict_dataset(row: dict[str, str], config_hash: str) -> dict:
    dataset = row["dataset"]
    try:
        oof_scores = _target_oof_scores(row)
        descriptor = scale_descriptor(
            build_descriptor(oof_scores), _DESCRIPTOR_CENTER, _DESCRIPTOR_SCALE
        )
        donors = select_donors(
            descriptor,
            _DONOR_DESCRIPTORS,
            _DONOR_COUNTS,
            minimum_reference=MIN_POOL_REFERENCE,
        )
        pooled_references = {
            source: np.concatenate([_DONOR_NORMALIZED[name][source] for name in donors])
            for source in SOURCES
        }

        train_raw = load_values_only(dataset, "TRAIN")
        test_raw = load_values_only(dataset, "TEST")
        fold_scores, target_length = score_fold(dataset, train_raw, test_raw)
        test_scores = {source: fold_scores[source][1] for source in SOURCES}
        normalized_oof, normalized_test = normalize_query_scores(oof_scores, test_scores)
        references = {
            "local_crossfit_reference": normalized_oof,
            "similarity_pool_reference": pooled_references,
        }
        rows = candidate_rows(
            dataset=dataset,
            n_train=len(train_raw),
            n_test=len(test_raw),
            normalized_test_scores=normalized_test,
            references=references,
            selected_donors=donors,
            config_hash=config_hash,
        )
        for prediction in rows:
            prediction["target_length"] = int(target_length)
            prediction["source_score_max"] = {
                source: float(np.max(normalized_test[source])) if normalized_test[source].size else None
                for source in SOURCES
            }
            prediction["source_oof_cache_sha256"] = row["file_hash"]
            assert_label_free_prediction(prediction)
        return {"dataset": dataset, "rows": rows, "error": None}
    except Exception as exc:
        return {
            "dataset": dataset,
            "rows": [],
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def target_and_donor_rows(dataset_limit: int | None = None):
    manifest_rows = validate_frozen_inputs()
    targets = [
        row for row in manifest_rows
        if row["status"] == "computed"
        and TARGET_MIN_TRAIN <= int(row["n_train"]) < TARGET_MAX_TRAIN_EXCLUSIVE
    ]
    donors = [
        row for row in manifest_rows
        if row["status"] == "computed" and int(row["n_train"]) >= DONOR_MIN_TRAIN
    ]
    if len(targets) != 316:
        raise ValueError(f"expected 316 targets, got {len(targets)}")
    if dataset_limit:
        targets = targets[: int(dataset_limit)]
    return targets, donors


def _checkpoint_rows(path: Path) -> dict[str, list[dict]]:
    completed: dict[str, list[dict]] = {}
    if not path.exists():
        return completed
    for item in read_jsonl(path):
        completed[item["dataset"]] = item["rows"]
    return completed


def run_prediction(output_dir: Path, workers: int, dataset_limit: int | None, reset: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    targets, donors = target_and_donor_rows(dataset_limit)
    normalized, descriptors, counts, center, scale = build_donor_pack(donors)
    config_hash = frozen_config_hash()
    checkpoint = output_dir / "prediction_checkpoint.jsonl"
    if reset and checkpoint.exists():
        checkpoint.unlink()
    completed = _checkpoint_rows(checkpoint)
    pending = [row for row in targets if row["dataset"] not in completed]
    errors: list[dict] = []
    progress = ExperimentProgress(
        experiment_id=EXPERIMENT_ID,
        label=LABEL,
        expected_datasets=len(targets),
        detail_csv=str(output_dir / "04_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective small-sample review-candidate validation",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.start(phase="label-free prediction")
    if completed:
        progress.update(done=len(completed), current_dataset=None, rows=len(completed) * 16, errors=0, phase="resume prediction")

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=initialize_worker,
        initargs=(normalized, descriptors, counts, center, scale),
    ) as executor:
        futures = {executor.submit(predict_dataset, row, config_hash): row["dataset"] for row in pending}
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
            progress.update(done=done, current_dataset=dataset, rows=done * 16, errors=len(errors), phase="label-free prediction")
            if done % 10 == 0 or done == len(targets):
                print(f"Exp166 prediction [{done}/{len(targets)}] errors={len(errors)}", flush=True)

    if errors:
        write_csv(output_dir / "prediction_errors.csv", errors)
        progress.fail(error=f"{len(errors)} prediction errors", done=len(completed), rows=len(completed) * 16, errors=len(errors))
        raise SystemExit(f"{len(errors)} prediction errors")
    expected = {row["dataset"] for row in targets}
    if set(completed) != expected:
        raise RuntimeError("prediction coverage incomplete")
    predictions = [row for name in sorted(completed) for row in completed[name]]
    predictions.sort(key=lambda row: (row["dataset"], row["method"], row["alpha"], row["agreement"]))
    prediction_path = output_dir / "02_frozen_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
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
    write_json(output_dir / "03_prediction_manifest.json", manifest)
    progress.update(done=len(targets), current_dataset=None, rows=len(predictions), errors=0, phase="prediction frozen; awaiting evaluation")
    return manifest


def run_evaluation(output_dir: Path) -> dict:
    manifest_path = output_dir / "03_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = output_dir / manifest["prediction_file"]
    actual_hash = sha256_file(prediction_path)
    if actual_hash != manifest["prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch")
    predictions = read_jsonl(prediction_path)
    labels_cache: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []
    for prediction in predictions:
        assert_label_free_prediction(prediction)
        dataset = prediction["dataset"]
        labels = labels_cache.setdefault(dataset, load_test_labels(dataset))
        if len(labels) != int(prediction["n_test"]):
            raise ValueError(f"TEST label length mismatch for {dataset}")
        metrics = evaluate_indices(prediction["candidate_indices"], labels)
        metric_rows.append(
            {
                "dataset": dataset,
                "config_name": f"{prediction['method']}::{prediction['agreement']}::alpha={prediction['alpha']}",
                "method": prediction["method"],
                "alpha": prediction["alpha"],
                "agreement": prediction["agreement"],
                "n_train": prediction["n_train"],
                "n_test": prediction["n_test"],
                "predicted_count": prediction["candidate_count"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "lane": "research_review_candidate",
                "metric_scope": "retrospective_human_review_candidate",
            }
        )
    write_csv(output_dir / "04_dataset_metrics.csv", metric_rows)

    summary_rows: list[dict] = []
    for method in METHODS:
        for alpha in ALPHAS:
            for agreement in ("2of3", "3of3"):
                group = [
                    row for row in metric_rows
                    if row["method"] == method and float(row["alpha"]) == alpha and row["agreement"] == agreement
                ]
                total_tp = sum(int(row["tp"]) for row in group)
                total_fp = sum(int(row["fp"]) for row in group)
                total_fn = sum(int(row["fn"]) for row in group)
                precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
                recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
                summary_rows.append(
                    {
                        "config_name": f"{method}::{agreement}::alpha={alpha}",
                        "method": method,
                        "alpha": alpha,
                        "agreement": agreement,
                        "datasets": len(group),
                        "review_candidates": sum(int(row["predicted_count"]) for row in group),
                        "review_tp": total_tp,
                        "review_fp": total_fp,
                        "review_fn": total_fn,
                        "review_precision": precision,
                        "review_recall": recall,
                        "mean_f1": float(np.mean([float(row["f1"]) for row in group])),
                        "median_f1": float(np.median([float(row["f1"]) for row in group])),
                        "zero_f1_datasets": sum(float(row["f1"]) == 0.0 for row in group),
                        "candidate_zero_datasets": sum(int(row["predicted_count"]) == 0 for row in group),
                        "mean_candidates_per_dataset": float(np.mean([int(row["predicted_count"]) for row in group])),
                        "autonomous_metric_scope": "not_evaluated",
                        "review_metric_scope": "retrospective_human_review_candidate",
                        "selected_as_primary": 0,
                        "retrospective": 1,
                    }
                )
    write_csv(output_dir / "05_global_summary.csv", summary_rows)

    transitions = []
    by_key = {(row["dataset"], float(row["alpha"]), row["agreement"], row["method"]): row for row in metric_rows}
    prediction_map = {(row["dataset"], float(row["alpha"]), row["agreement"], row["method"]): row for row in predictions}
    for dataset in sorted(labels_cache):
        for alpha in ALPHAS:
            for agreement in ("2of3", "3of3"):
                local = prediction_map[(dataset, alpha, agreement, "local_crossfit_reference")]
                pooled = prediction_map[(dataset, alpha, agreement, "similarity_pool_reference")]
                local_set = set(local["candidate_indices"])
                pooled_set = set(pooled["candidate_indices"])
                union = local_set | pooled_set
                transitions.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha,
                        "agreement": agreement,
                        "local_count": len(local_set),
                        "pooled_count": len(pooled_set),
                        "added_by_pool": len(pooled_set - local_set),
                        "removed_by_pool": len(local_set - pooled_set),
                        "exact_jaccard": len(local_set & pooled_set) / len(union) if union else 1.0,
                        "local_f1": by_key[(dataset, alpha, agreement, "local_crossfit_reference")]["f1"],
                        "pooled_f1": by_key[(dataset, alpha, agreement, "similarity_pool_reference")]["f1"],
                    }
                )
    write_csv(output_dir / "06_candidate_transition.csv", transitions)
    result = {
        "errors": 0,
        "datasets": len(labels_cache),
        "prediction_rows": len(predictions),
        "evaluation_rows": len(metric_rows),
        "summary_rows": len(summary_rows),
        "prediction_sha256": actual_hash,
        "selected_primary_policy": None,
        "retrospective": True,
    }
    write_json(output_dir / "07_completion.json", result)
    progress = ExperimentProgress(
        experiment_id=EXPERIMENT_ID,
        label=LABEL,
        expected_datasets=len(labels_cache),
        detail_csv=str(output_dir / "04_dataset_metrics.csv"),
        stdout_log=str(output_dir / "full_run.log"),
        scope="retrospective small-sample review-candidate validation",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.finish(rows=len(metric_rows), result_summary=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "config_hash": frozen_config_hash(),
        "scope": "316 datasets with 5 <= n_train < 20",
        "alphas": ALPHAS,
        "agreements": AGREEMENTS,
        "methods": METHODS,
        "prediction_uses_test_values": True,
        "prediction_uses_test_labels": False,
        "test_metric_policy_selection": False,
        "candidate_lane": "research_review_candidate",
        "retrospective": True,
    }
    write_json(args.output_dir / "01_contract.json", contract)
    if args.phase in {"predict", "all"}:
        run_prediction(args.output_dir, args.workers, args.dataset_limit, args.reset_checkpoint)
    if args.phase in {"evaluate", "all"}:
        result = run_evaluation(args.output_dir)
        print(f"{EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
