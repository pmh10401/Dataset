from __future__ import annotations

import os
import json
import tempfile
from pathlib import Path
import numpy as np

REQUIRED_SCORE_ARRAYS = (
    "spectral_train", "spectral_test",
    "shapelet_train", "shapelet_test",
    "interval_train", "interval_test",
)
FORBIDDEN_METADATA_KEYS = {
    "tp", "fp", "fn", "f1", "anomaly_count", "test_label",
    "test_labels", "candidate_index", "candidate_indices", "threshold", "lane",
}

def _has_forbidden_metadata(metadata: dict) -> bool:
    normalized = {str(key).strip().lower().replace(" ", "_") for key in metadata}
    return bool(normalized & FORBIDDEN_METADATA_KEYS)

def write_cache_atomic(path: str | Path, payload: dict[str, np.ndarray], metadata: dict) -> None:
    """
    Writes the cache payload and metadata to a compressed NPZ file atomically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # We check forbidden keys before writing
    if _has_forbidden_metadata(metadata):
        raise ValueError("Forbidden evaluation key found in metadata")
    for name in REQUIRED_SCORE_ARRAYS:
        if name not in payload:
            raise ValueError(f"Missing score array: {name}")
        if np.asarray(payload[name]).ndim != 1 or not np.all(np.isfinite(payload[name])):
            raise ValueError(f"Invalid score array: {name}")

    # Create temporary file in the same directory to ensure atomic move on same filesystem
    fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    try:
        np.savez_compressed(
            tmp_path,
            spectral_train=payload["spectral_train"],
            spectral_test=payload["spectral_test"],
            shapelet_train=payload["shapelet_train"],
            shapelet_test=payload["shapelet_test"],
            interval_train=payload["interval_train"],
            interval_test=payload["interval_test"],
            metadata_json=np.array(json.dumps(metadata))
        )
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e

def validate_cache(path: str | Path, expected_config_hash: str, expected_dataset: str | None = None) -> bool:
    """
    Validates the cache file format, finiteness of arrays, config hash match,
    required metadata keys, and lack of forbidden keys.
    Returns True if valid, False otherwise.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        # np.load might fail if file is corrupted
        with np.load(path, allow_pickle=False) as data:
            required_arrays = {
                "spectral_train", "spectral_test",
                "shapelet_train", "shapelet_test",
                "interval_train", "interval_test",
                "metadata_json"
            }
            if not required_arrays.issubset(data.files):
                return False

            # Verify arrays are finite
            for name in REQUIRED_SCORE_ARRAYS:
                arr = data[name]
                if not np.all(np.isfinite(arr)):
                    return False
                if arr.ndim != 1:
                    return False

            # Parse and validate metadata
            meta_json = data["metadata_json"]
            meta_str = str(meta_json) if meta_json.ndim == 0 else str(meta_json[0])
            metadata = json.loads(meta_str)

            required_keys = {
                "dataset", "target_length", "source_config_hash",
                "seed", "train_rows", "test_rows", "contains_test_labels"
            }
            if not required_keys.issubset(metadata.keys()):
                return False

            if metadata["source_config_hash"] != expected_config_hash:
                return False

            if expected_dataset is not None and metadata["dataset"] != expected_dataset:
                return False

            if metadata["contains_test_labels"] is not False:
                return False

            # Forbidden keys check
            if _has_forbidden_metadata(metadata):
                return False

            train_rows = int(metadata["train_rows"])
            test_rows = int(metadata["test_rows"])
            if train_rows <= 0 or test_rows <= 0:
                return False
            if any(len(data[name]) != train_rows for name in ("spectral_train", "shapelet_train", "interval_train")):
                return False
            if any(len(data[name]) != test_rows for name in ("spectral_test", "shapelet_test", "interval_test")):
                return False

        return True
    except Exception:
        return False
