from __future__ import annotations

import os
import sqlite3
import numpy as np
from experiments.exp162.full_coverage_score_core import (
    spectral_features,
    shapelet_features,
    interval_features,
    robust_score_pair,
    source_config_hash,
    stable_hash,
    resample_rows,
)

DB_PATH = "/Users/minho/Documents/Dataset/univariate_ts.db"

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

def load_dataset_series_only(dataset_name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Loads train and test series from the SQLite database.
    Does NOT select labels or anomaly positions.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
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

def build_dataset_cache(dataset_name: str) -> tuple[dict[str, np.ndarray], dict]:
    """
    Builds the payload and metadata for the score cache of a dataset.
    """
    X_train, X_test, target_len = load_dataset_series_only(dataset_name)

    # 1. Spectral features
    spectral_train_feat = spectral_features(X_train)
    spectral_test_feat = spectral_features(X_test)
    spec_tr, spec_te = robust_score_pair(spectral_train_feat, spectral_test_feat)

    # 2. Shapelet features
    shap_train_feat, shap_test_feat = shapelet_features(dataset_name, X_train, X_test, target_len)
    shap_tr, shap_te = robust_score_pair(shap_train_feat, shap_test_feat)

    # 3. Interval features
    interval_train_feat = interval_features(X_train)
    interval_test_feat = interval_features(X_test)
    int_tr, int_te = robust_score_pair(interval_train_feat, interval_test_feat)

    payload = {
        "spectral_train": spec_tr.astype(np.float32),
        "spectral_test": spec_te.astype(np.float32),
        "shapelet_train": shap_tr.astype(np.float32),
        "shapelet_test": shap_te.astype(np.float32),
        "interval_train": int_tr.astype(np.float32),
        "interval_test": int_te.astype(np.float32),
    }

    metadata = {
        "dataset": dataset_name,
        "target_length": int(target_len),
        "source_config_hash": source_config_hash(),
        "seed": int(20260714 + stable_hash(dataset_name)),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "contains_test_labels": False
    }

    # Forbidden key check
    forbidden_keys = {"TP", "FP", "FN", "F1", "anomaly_count", "TEST label", "candidate_index", "threshold", "lane"}
    for key in forbidden_keys:
        if key in metadata:
            raise ValueError(f"Forbidden key '{key}' found in metadata")

    return payload, metadata
