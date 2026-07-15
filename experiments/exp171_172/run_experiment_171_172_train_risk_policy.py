#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from experiment_progress import ExperimentProgress
from experiments.exp167_168.parity_core import evaluate_indices
from experiments.exp171_172.policy_core import select_train_risk_policy

AUDIT_EXPERIMENT_ID = "experiment_171_train_normal_risk_policy_freeze"
EVALUATION_EXPERIMENT_ID = "experiment_172_frozen_train_risk_policy_review"
AUDIT_LABEL = "Exp171 · TRAIN-normal risk policy freeze"
EVALUATION_LABEL = "Exp172 · Frozen review policy evaluation"
DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DB_SHA256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
SOURCE_DIR = PROJECT_DIR / "outputs/exp169_170_oob_pooled_calibration"
SOURCE_MANIFEST = SOURCE_DIR / "MANIFEST.csv"
SOURCE_DATASET_AUDIT = SOURCE_DIR / "04_exp169_dataset_audit.csv"
SOURCE_GLOBAL_AUDIT = SOURCE_DIR / "05_exp169_global_summary.csv"
SOURCE_PREDICTIONS = SOURCE_DIR / "06_frozen_predictions.jsonl"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs/exp171_172_train_risk_policy"
EXPECTED_DATASETS = 316
RISK_TARGET = 0.01
CONFIDENCE = 0.95
CONFIG_VERSION = "exp171_172_train_risk_policy_v1"

FORBIDDEN_PREDICTION_FIELDS = {
    "y_test",
    "label",
    "labels",
    "actual_anomaly_positions",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "combined_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0]) if rows else [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def source_manifest_map() -> dict[str, dict]:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)
    rows = read_csv(SOURCE_MANIFEST)
    mapping = {row["file"]: row for row in rows}
    for path in (SOURCE_DATASET_AUDIT, SOURCE_GLOBAL_AUDIT, SOURCE_PREDICTIONS):
        row = mapping.get(path.name)
        if row is None:
            raise ValueError(f"source manifest lacks {path.name}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise ValueError(f"source hash mismatch for {path.name}: {actual}")
    return mapping


def frozen_contract() -> dict:
    source_manifest_map()
    config = {
        "version": CONFIG_VERSION,
        "risk_target": RISK_TARGET,
        "confidence": CONFIDENCE,
        "multiple_comparison_correction": "bonferroni",
        "comparisons": 60,
        "selection_rule": "max_train_oob_review_count_under_simultaneous_risk_cap",
        "candidate_lane": "research_review_candidate",
        "autonomous_hard_alert": False,
        "test_metric_selection": False,
        "family_routing": False,
        "dataset_name_routing": False,
        "test_length_budget": False,
    }
    return {
        "audit_experiment_id": AUDIT_EXPERIMENT_ID,
        "evaluation_experiment_id": EVALUATION_EXPERIMENT_ID,
        "config": config,
        "config_hash": stable_hash(config),
        "expected_datasets": EXPECTED_DATASETS,
        "database_path": str(DB_PATH),
        "database_sha256_expected": DB_SHA256,
        "source_files": {
            path.name: sha256_file(path)
            for path in (SOURCE_MANIFEST, SOURCE_DATASET_AUDIT, SOURCE_GLOBAL_AUDIT, SOURCE_PREDICTIONS)
        },
        "selection_input_scope": "train_oob_only",
        "prediction_uses_test_labels": False,
        "retrospective": True,
        "prospective_validated": False,
        "end_to_end_train_only_provenance_validated": False,
    }


def assert_label_free_selected_prediction(row: dict) -> None:
    present = FORBIDDEN_PREDICTION_FIELDS.intersection(row)
    if present:
        raise ValueError(f"prediction contains forbidden evaluation fields: {sorted(present)}")
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    indices = [int(index) for index in row.get("candidate_indices", [])]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("candidate indices must be unique and sorted")
    if any(index < 0 or index >= int(row["n_test"]) for index in indices):
        raise ValueError("candidate index out of range")
    if len(indices) != int(row["candidate_count"]):
        raise ValueError("candidate count mismatch")


def run_audit(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = frozen_contract()
    write_json(output_dir / "01_contract.json", contract)
    global_rows = read_csv(SOURCE_GLOBAL_AUDIT)
    dataset_rows = read_csv(SOURCE_DATASET_AUDIT)
    result = select_train_risk_policy(
        global_rows,
        dataset_rows,
        risk_target=RISK_TARGET,
        confidence=CONFIDENCE,
        expected_dataset_count=EXPECTED_DATASETS,
    )
    candidate_rows = sorted(
        result["candidates"],
        key=lambda row: (row["pool_policy"], float(row["alpha"]), row["source_config"]),
    )
    write_csv(output_dir / "02_exp171_policy_candidates.csv", candidate_rows)
    policy = {
        key: value for key, value in result.items() if key != "candidates"
    }
    policy.update(
        {
            "experiment_id": AUDIT_EXPERIMENT_ID,
            "config_hash": contract["config_hash"],
            "source_dataset_audit_sha256": sha256_file(SOURCE_DATASET_AUDIT),
            "source_global_audit_sha256": sha256_file(SOURCE_GLOBAL_AUDIT),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "retrospective": True,
            "prospective_validated": False,
        }
    )
    write_json(output_dir / "03_exp171_frozen_policy.json", policy)
    progress = ExperimentProgress(
        experiment_id=AUDIT_EXPERIMENT_ID,
        label=AUDIT_LABEL,
        expected_datasets=EXPECTED_DATASETS,
        detail_csv=str(output_dir / "02_exp171_policy_candidates.csv"),
        scope="TRAIN-normal OOB policy freeze",
        autonomous_metric="not_evaluated",
        review_metric="normal_review_risk_only",
    )
    progress.start(phase="TRAIN-only policy selection")
    progress.finish(
        rows=len(candidate_rows),
        result_summary={
            "status": policy["status"],
            "selected": policy["selected"],
            "test_metrics_used": 0,
        },
    )
    return policy


def _load_frozen_policy(output_dir: Path) -> tuple[dict, Path, str]:
    contract_path = output_dir / "01_contract.json"
    policy_path = output_dir / "03_exp171_frozen_policy.json"
    if not contract_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError("Exp171 contract or frozen policy is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("config_hash") != contract.get("config_hash"):
        raise ValueError("frozen policy config hash mismatch")
    if policy.get("selected") is None:
        raise ValueError("Exp171 abstained; Exp172 prediction is forbidden")
    source_manifest_map()
    if policy["source_dataset_audit_sha256"] != sha256_file(SOURCE_DATASET_AUDIT):
        raise ValueError("frozen policy dataset-audit hash mismatch")
    if policy["source_global_audit_sha256"] != sha256_file(SOURCE_GLOBAL_AUDIT):
        raise ValueError("frozen policy global-audit hash mismatch")
    return policy, policy_path, sha256_file(policy_path)


def run_prediction(output_dir: Path, dataset_limit: int | None = None) -> dict:
    policy, policy_path, policy_hash = _load_frozen_policy(output_dir)
    selected_key = tuple(policy["selected"]["policy_key"])
    source_rows = read_jsonl(SOURCE_PREDICTIONS)
    selected_rows = [
        row
        for row in source_rows
        if (
            row["pool_policy"],
            float(row["alpha"]),
            row["source_config"],
        )
        == (selected_key[0], float(selected_key[1]), selected_key[2])
    ]
    selected_rows.sort(key=lambda row: row["dataset"])
    if len(selected_rows) != EXPECTED_DATASETS:
        raise ValueError(f"selected prediction coverage mismatch: {len(selected_rows)}")
    if len({row["dataset"] for row in selected_rows}) != EXPECTED_DATASETS:
        raise ValueError("selected prediction has duplicate datasets")
    if dataset_limit is not None:
        selected_rows = selected_rows[: int(dataset_limit)]

    predictions = []
    for source_row in selected_rows:
        row = dict(source_row)
        row.update(
            {
                "experiment_id": EVALUATION_EXPERIMENT_ID,
                "selected_policy_sha256": policy_hash,
                "selection_input_scope": "train_oob_only",
                "test_metrics_used_for_selection": 0,
                "lane": "research_review_candidate",
            }
        )
        assert_label_free_selected_prediction(row)
        predictions.append(row)

    prediction_path = output_dir / "04_exp172_frozen_predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    manifest = {
        "experiment_id": EVALUATION_EXPERIMENT_ID,
        "policy_file": policy_path.name,
        "policy_sha256": policy_hash,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "source_prediction_file": SOURCE_PREDICTIONS.name,
        "source_prediction_sha256": sha256_file(SOURCE_PREDICTIONS),
        "dataset_count": len(predictions),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "autonomous_metric_scope": "not_evaluated",
        "review_metric_scope": "retrospective_research_review_candidate",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "05_prediction_manifest.json", manifest)
    return manifest


def verify_prediction_manifest(output_dir: Path) -> tuple[dict, Path, dict]:
    manifest_path = output_dir / "05_prediction_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy_path = output_dir / manifest["policy_file"]
    prediction_path = output_dir / manifest["prediction_file"]
    if sha256_file(policy_path) != manifest["policy_sha256"]:
        raise ValueError("frozen policy hash mismatch")
    if sha256_file(prediction_path) != manifest["prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    return manifest, prediction_path, policy


def _verify_database_hash() -> str:
    if not DB_PATH.is_file():
        raise FileNotFoundError(DB_PATH)
    actual = sha256_file(DB_PATH)
    if actual != DB_SHA256:
        raise ValueError(f"database hash mismatch: {actual}")
    return actual


def _load_test_labels(dataset: str) -> np.ndarray:
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


def run_evaluation(output_dir: Path) -> dict:
    manifest, prediction_path, policy = verify_prediction_manifest(output_dir)
    _verify_database_hash()
    predictions = read_jsonl(prediction_path)
    if len(predictions) != int(manifest["dataset_count"]):
        raise ValueError("prediction row count mismatch")
    if len({row["dataset"] for row in predictions}) != len(predictions):
        raise ValueError("prediction dataset coverage contains duplicates")

    metric_rows = []
    for prediction in predictions:
        assert_label_free_selected_prediction(prediction)
        labels = _load_test_labels(prediction["dataset"])
        if labels.size != int(prediction["n_test"]):
            raise ValueError(f"TEST length mismatch for {prediction['dataset']}")
        metrics = evaluate_indices(prediction["candidate_indices"], labels)
        metric_rows.append(
            {
                "dataset": prediction["dataset"],
                "pool_policy": prediction["pool_policy"],
                "alpha": prediction["alpha"],
                "source_config": prediction["source_config"],
                "n_train": prediction["n_train"],
                "n_test": prediction["n_test"],
                "review_candidates": prediction["candidate_count"],
                **metrics,
                "autonomous_metric_scope": "not_evaluated",
                "review_metric_scope": "retrospective_research_review_candidate",
            }
        )
    metric_rows.sort(key=lambda row: row["dataset"])
    write_csv(output_dir / "06_exp172_dataset_metrics.csv", metric_rows)

    tp = sum(int(row["tp"]) for row in metric_rows)
    fp = sum(int(row["fp"]) for row in metric_rows)
    summary = {
        "pool_policy": policy["selected"]["pool_policy"],
        "alpha": policy["selected"]["alpha"],
        "source_config": policy["selected"]["source_config"],
        "datasets": len(metric_rows),
        "review_candidates": sum(int(row["review_candidates"]) for row in metric_rows),
        "tp": tp,
        "fp": fp,
        "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
        "mean_recall": float(np.mean([float(row["recall"]) for row in metric_rows])),
        "mean_f1": float(np.mean([float(row["f1"]) for row in metric_rows])),
        "zero_f1_datasets": sum(float(row["f1"]) == 0.0 for row in metric_rows),
        "candidate_zero_datasets": sum(int(row["review_candidates"]) == 0 for row in metric_rows),
        "mean_candidates_per_dataset": float(
            np.mean([int(row["review_candidates"]) for row in metric_rows])
        ),
        "autonomous_metric_scope": "not_evaluated",
        "review_metric_scope": "retrospective_research_review_candidate",
        "policy_selected_from_test_metrics": 0,
        "retrospective": 1,
    }
    write_csv(output_dir / "07_exp172_global_summary.csv", [summary])
    completion = {
        "experiment_id": EVALUATION_EXPERIMENT_ID,
        "dataset_count": len(metric_rows),
        "prediction_sha256": manifest["prediction_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "selected_policy": policy["selected"],
        "autonomous_metric_scope": "not_evaluated",
        "review_metric_scope": "retrospective_research_review_candidate",
        "retrospective": True,
        "prospective_validated": False,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(output_dir / "10_completion.json", completion)

    progress = ExperimentProgress(
        experiment_id=EVALUATION_EXPERIMENT_ID,
        label=EVALUATION_LABEL,
        expected_datasets=len(metric_rows),
        detail_csv=str(output_dir / "06_exp172_dataset_metrics.csv"),
        scope="frozen retrospective research review policy",
        autonomous_metric="not_evaluated",
        review_metric="candidate_precision_recall_f1",
    )
    progress.start(phase="hash-verified retrospective evaluation")
    progress.finish(rows=len(metric_rows), result_summary=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("audit", "predict", "evaluate", "predict-evaluate", "all"),
        default="all",
    )
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"audit", "all"}:
        result = run_audit(args.output_dir)
        print(f"{AUDIT_EXPERIMENT_ID} finished: {json.dumps(result['selected'], sort_keys=True)}")
    if args.phase in {"predict", "predict-evaluate", "all"}:
        result = run_prediction(args.output_dir, args.dataset_limit)
        print(f"prediction frozen: {json.dumps(result, sort_keys=True)}")
    if args.phase in {"evaluate", "predict-evaluate", "all"}:
        result = run_evaluation(args.output_dir)
        print(f"{EVALUATION_EXPERIMENT_ID} finished: {json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
