from __future__ import annotations

import csv
import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from experiments.exp162.cache_io import validate_cache


FORBIDDEN_PREDICTION_KEYS = {
    "y_test",
    "test_label",
    "test_labels",
    "hard_tp",
    "hard_fp",
    "hard_fn",
    "hard_precision",
    "hard_recall",
    "hard_f1",
    "standard_review_tp",
    "standard_review_fp",
    "standard_review_fn",
    "standard_review_precision",
    "standard_review_recall",
    "standard_review_f1",
    "priority_review_tp",
    "priority_review_fp",
    "priority_review_fn",
    "priority_review_precision",
    "priority_review_recall",
    "priority_review_f1",
    "combined_tp",
    "combined_fp",
    "combined_fn",
    "combined_precision",
    "combined_recall",
    "combined_f1",
    "mean_combined_f1",
}


def get_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_file_hash(path: Path, expected_hash: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    actual_hash = get_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"File hash mismatch for {path.name}!\nExpected: {expected_hash}\nActual:   {actual_hash}"
        )


def load_exp160_predictions(path: Path, expected_hash: str) -> list[dict]:
    verify_file_hash(path, expected_hash)
    predictions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) & FORBIDDEN_PREDICTION_KEYS:
                raise ValueError("Exp160 prediction unexpectedly contains evaluation fields")
            predictions.append(row)
    return predictions


def load_exp162_manifest(path: Path, expected_hash: str) -> dict[str, dict]:
    verify_file_hash(path, expected_hash)
    manifest = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            if dataset in manifest:
                raise ValueError(f"duplicate dataset in Exp162 manifest: {dataset}")
            manifest[dataset] = {
                "file_path": row["file_path"],
                "file_hash": row["file_hash"],
                "status": row["status"],
            }
    return manifest


def load_exp162_contract(path: Path, expected_hash: str) -> dict:
    verify_file_hash(path, expected_hash)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_scores(
    dataset_name: str,
    manifest_entry: dict,
    expected_config_hash: str,
    project_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Validate and load dataset alternative scores from Exp162 cache.
    """
    file_rel_path = manifest_entry["file_path"]
    expected_file_hash = manifest_entry["file_hash"]
    
    file_path = project_dir / file_rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Cache file for {dataset_name} not found at {file_path}")
        
    # 1. File-level SHA256 integrity check
    verify_file_hash(file_path, expected_file_hash)
    
    # 2. Structural/Semantic cache validation (finiteness, config hash match, etc.)
    if not validate_cache(file_path, expected_config_hash, expected_dataset=dataset_name):
        raise ValueError(f"Cache validation failed for dataset {dataset_name} at {file_rel_path}")
        
    # 3. Load arrays
    with np.load(file_path, allow_pickle=False) as data:
        scores = {
            "spectral_train": data["spectral_train"],
            "spectral_test": data["spectral_test"],
            "shapelet_train": data["shapelet_train"],
            "shapelet_test": data["shapelet_test"],
            "interval_train": data["interval_train"],
            "interval_test": data["interval_test"],
        }
        
    # Ensure they are indeed finite (validate_cache already checked, but for redundancy)
    for k, v in scores.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"Score array {k} contains non-finite values for {dataset_name}")
            
    return scores


def write_predictions_atomic(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    for row in predictions:
        forbidden = sorted(set(row) & FORBIDDEN_PREDICTION_KEYS)
        if forbidden:
            raise ValueError(f"forbidden prediction keys: {forbidden}")
    
    fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp.jsonl", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp.json", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)
    try:
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def freeze_prediction_file(
    prediction_path: Path,
    manifest_path: Path,
    *,
    row_count: int,
    input_hashes: dict[str, str],
    config_hash: str,
) -> dict:
    """Freeze label-free predictions with their exact inputs and config."""

    actual_rows = sum(
        1
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if actual_rows != int(row_count):
        raise ValueError(
            f"prediction row count mismatch expected={row_count} actual={actual_rows}"
        )
    payload = {
        "prediction_file": str(prediction_path),
        "prediction_sha256": get_sha256(prediction_path),
        "row_count": actual_rows,
        "input_hashes": dict(sorted(input_hashes.items())),
        "config_hash": str(config_hash),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_path, payload)
    return payload


def verify_prediction_manifest(
    prediction_path: Path,
    manifest_path: Path,
    *,
    expected_row_count: int | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_file_hash(prediction_path, str(manifest["prediction_sha256"]))
    actual_rows = sum(
        1
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    manifest_rows = int(manifest["row_count"])
    if actual_rows != manifest_rows:
        raise ValueError("prediction row count does not match frozen manifest")
    if expected_row_count is not None and actual_rows != int(expected_row_count):
        raise ValueError(
            f"prediction row count mismatch expected={expected_row_count} actual={actual_rows}"
        )
    return manifest
