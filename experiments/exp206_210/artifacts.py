from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.exp206_210.core import (
    ALPHAS,
    SEED,
    validate_prediction_payload,
    sha256_file,
)

DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"
EXP137_CSV = DATA_DIR / "experiment_137_operational_triage_results.csv"

# Exp153 locations
EXP153_ROOT = Path("/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements")
EXP153_MANIFEST = EXP153_ROOT / "01_score_cache_manifest.csv"
EXP153_CACHE_DIR = EXP153_ROOT / "cache"

# Exp206 output locations
EXP206_ROOT = DATA_DIR / "exp206_210_residual_coverage"
EXP206_CACHE_DIR = EXP206_ROOT / "cache"
EXP206_MANIFEST = EXP206_ROOT / "exp206_cache_manifest.json"


@dataclass(frozen=True)
class FrozenInputs:
    datasets: list[dict[str, object]]
    input_hashes: dict[str, str]


def sanitize_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite_mask = np.isfinite(x)
    if finite_mask.all():
        return x
    if not finite_mask.any():
        return np.zeros_like(x, dtype=np.float32)
    idx = np.arange(len(x))
    cleaned = x.copy()
    cleaned[~finite_mask] = np.interp(idx[~finite_mask], idx[finite_mask], x[finite_mask])
    return cleaned.astype(np.float32)


def resample_rows(X: np.ndarray, target_len: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("resample_rows expects a 2D array")
    target_len = max(8, int(target_len))
    if X.shape[1] == target_len:
        return X.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.vstack([np.interp(new_x, old_x, row).astype(np.float32) for row in X])


def load_dataset_series_only(dataset_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Loads train and test series from the SQLite database.
    Does NOT select labels or anomaly positions.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT series_length FROM datasets WHERE name = ?", (dataset_name,))
        meta = cursor.fetchone()
        if not meta:
            raise ValueError(f"Dataset {dataset_name} not found in database")

        cursor.execute(
            """
            SELECT i.values_blob
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TRAIN'
            ORDER BY i.instance_index
            """,
            (dataset_name,),
        )
        train_rows = cursor.fetchall()
        if not train_rows:
            raise ValueError(f"No TRAIN instances found for dataset {dataset_name}")

        cursor.execute(
            """
            SELECT i.values_blob
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset_name,),
        )
        test_rows = cursor.fetchall()
        if not test_rows:
            raise ValueError(f"No TEST instances found for dataset {dataset_name}")
    finally:
        conn.close()

    X_train_raw = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in train_rows]
    X_test_raw = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in test_rows]

    # Target length: median TRAIN-normal series length, clipped to [8, 2048]
    lengths = [len(x) for x in X_train_raw]
    target_len = int(np.clip(np.median(lengths), 8, 2048))

    X_train = np.vstack([
        resample_rows(series.reshape(1, -1), target_len)[0]
        for series in X_train_raw
    ]).astype(np.float32)
    X_test = np.vstack([
        resample_rows(series.reshape(1, -1), target_len)[0]
        for series in X_test_raw
    ]).astype(np.float32)

    return X_train, X_test, target_len


def load_population_values_only(limit: int | None = None) -> list[dict[str, object]]:
    """
    Selects exactly 251 sorted datasets with train_normal_count >= 100.
    """
    if not EXP137_CSV.exists():
        raise FileNotFoundError(f"Exp137 baseline not found at {EXP137_CSV}")

    allowed = []
    with EXP137_CSV.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            name = raw["dataset_name"]
            n_train = int(raw["train_normal_count"])
            n_test = int(raw["test_size"])
            # Extract Exp137 lane indices
            def parse_indices(text: str) -> set[int]:
                text = text.replace(",", " ").strip()
                return {int(tok) for tok in text.split() if tok}

            if n_train >= 100:
                allowed.append({
                    "dataset": name,
                    "n_train": n_train,
                    "n_test": n_test,
                    "hard_indices": parse_indices(raw["hard_alert_indices"]),
                    "standard_indices": parse_indices(raw["standard_review_indices"]),
                    "priority_indices": parse_indices(raw["priority_review_indices"]),
                })

    allowed.sort(key=lambda x: x["dataset"])
    if len(allowed) != 251:
        raise ValueError(f"Expected 251 datasets with train_normal_count >= 100, found {len(allowed)}")

    if limit is not None:
        allowed = allowed[:limit]
    return allowed


def load_current_oof_cache(dataset: str) -> dict[str, np.ndarray]:
    """
    Loads current four-source OOF cache from Exp153.
    """
    if not EXP153_MANIFEST.exists():
        raise FileNotFoundError(f"Exp153 manifest not found at {EXP153_MANIFEST}")

    with EXP153_MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["dataset_name"] == dataset:
                cache_file = row["cache_file"]
                cache_path = EXP153_CACHE_DIR / cache_file
                if not cache_path.exists():
                    raise FileNotFoundError(f"Exp153 cache file not found: {cache_path}")
                # Verify SHA256
                expected_sha = row["cache_sha256"]
                actual_sha = sha256_file(cache_path)
                if actual_sha != expected_sha:
                    raise ValueError(f"Exp153 cache SHA256 mismatch for {dataset}")

                with np.load(cache_path, allow_pickle=False) as archive:
                    result = {}
                    for source in (
                        "rocket_exp40", "exp55_best", "exp56_best", "exp84"
                    ):
                        result[f"{source}_train"] = np.asarray(
                            archive[f"{source}__train_oof"], dtype=np.float32
                        )
                        result[f"{source}_test"] = np.asarray(
                            archive[f"{source}__test"], dtype=np.float32
                        )
                    return result
    raise ValueError(f"Dataset {dataset} not found in Exp153 manifest")


def load_exp206_cache(
    dataset: str, cache_dir: Path | None = None
) -> dict[str, np.ndarray]:
    """
    Loads alternative OOF cache NPZ file for dataset from Exp206.
    """
    cache_path = Path(cache_dir or EXP206_CACHE_DIR) / f"{dataset}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Exp206 cache not found for dataset {dataset}")
    with np.load(cache_path, allow_pickle=False) as archive:
        return {
            f"{src}_{split}": np.asarray(archive[f"{src}_{split}"], dtype=np.float32)
            for src in ["spectral", "shapelet", "interval"]
            for split in ["train", "test"]
        }


def load_test_labels_for_evaluation(dataset: str) -> np.ndarray:
    """
    Loads test labels from the SQLite database.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT CAST(i.label AS INTEGER)
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No TEST instances found for dataset {dataset}")

    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)


def freeze_predictions(
    stage: str,
    experiment_id: str,
    rows: list[dict[str, object]],
    output_root: Path,
    input_hashes: dict[str, str],
) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / f"exp{stage}_frozen_predictions.jsonl"
    manifest_path = output_root / f"exp{stage}_prediction_manifest.json"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            validate_prediction_payload(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "experiment_id": experiment_id,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": len(rows),
        "dataset_count": len({str(row["dataset"]) for row in rows}),
        "input_hashes": dict(sorted(input_hashes.items())),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return prediction_path, manifest_path


def load_verified_predictions(
    prediction_path: Path, manifest_path: Path
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    expected = str(manifest["prediction_sha256"])
    actual = sha256_file(Path(prediction_path))
    if actual != expected:
        raise ValueError(
            f"prediction hash mismatch expected={expected} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        validate_prediction_payload(row)
    if len(rows) != int(manifest["prediction_rows"]):
        raise ValueError("prediction row count mismatch")
    return rows
