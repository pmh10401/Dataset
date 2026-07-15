from __future__ import annotations
import os
import sqlite3
import numpy as np

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

def load_train_normal_raw(dataset_name: str) -> list[np.ndarray]:
    """
    Loads train normal series from the SQLite database.
    Strictly queries TRAIN split values_blob without TEST or labels.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
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
        rows = cursor.fetchall()
        if not rows:
            raise ValueError(f"No TRAIN instances found for dataset {dataset_name}")
    finally:
        conn.close()

    raw_series = [sanitize_series(np.frombuffer(row[0], dtype=np.float32)) for row in rows]
    return raw_series
