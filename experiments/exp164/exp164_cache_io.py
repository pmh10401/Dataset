from __future__ import annotations

import os
import json
import hashlib
import tempfile
from pathlib import Path
import numpy as np

REQUIRED_ARRAYS = (
    "query_score_spectral",
    "query_score_shapelet",
    "query_score_interval",
    "fold_indices",
    "pseudo_run_indices",
    "fit_counts",
    "target_lengths",
    "thresholds_spectral",
    "thresholds_shapelet",
    "thresholds_interval",
    "flags_spectral",
    "flags_shapelet",
    "flags_interval",
    "agreement_2of3",
    "agreement_3of3",
)

FORBIDDEN_METADATA_KEYS = {
    "tp", "fp", "fn", "f1", "anomaly_count", "test_label", "test_labels", "test_values",
    "candidate_index", "candidate_indices", "threshold", "lane"
}

def calculate_sha256(filepath: str | Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()

def _has_forbidden_metadata(metadata: dict) -> bool:
    normalized = {str(key).strip().lower().replace(" ", "_") for key in metadata}
    return bool(normalized & FORBIDDEN_METADATA_KEYS)

def write_cache_atomic(path: str | Path, payload: dict[str, np.ndarray], metadata: dict) -> str:
    """
    Writes payload and metadata atomically using a temporary file.
    Returns the SHA256 of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validation checks before write
    if _has_forbidden_metadata(metadata):
        raise ValueError("Forbidden evaluation key found in metadata")

    for key in ("contains_test_values", "contains_test_labels", "contains_evaluation_metrics"):
        if metadata.get(key) is not False:
            raise ValueError(f"Metadata field '{key}' must be False")

    for name in REQUIRED_ARRAYS:
        if name not in payload:
            raise ValueError(f"Missing required array in cache payload: {name}")
        arr = np.asarray(payload[name])
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Array '{name}' has non-finite values")

    fd, tmp_path_str = tempfile.mkstemp(suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_path_str)

    try:
        np.savez_compressed(
            tmp_path,
            query_score_spectral=payload["query_score_spectral"],
            query_score_shapelet=payload["query_score_shapelet"],
            query_score_interval=payload["query_score_interval"],
            fold_indices=payload["fold_indices"],
            pseudo_run_indices=payload["pseudo_run_indices"],
            fit_counts=payload["fit_counts"],
            target_lengths=payload["target_lengths"],
            thresholds_spectral=payload["thresholds_spectral"],
            thresholds_shapelet=payload["thresholds_shapelet"],
            thresholds_interval=payload["thresholds_interval"],
            flags_spectral=payload["flags_spectral"],
            flags_shapelet=payload["flags_shapelet"],
            flags_interval=payload["flags_interval"],
            agreement_2of3=payload["agreement_2of3"],
            agreement_3of3=payload["agreement_3of3"],
            metadata_json=np.array(json.dumps(metadata))
        )
        os.replace(tmp_path, path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise e

    return calculate_sha256(path)

def validate_cache(path: str | Path, expected_config_hash: str, expected_dataset: str | None = None) -> tuple[bool, str]:
    """
    Validates cache files for format, finiteness, config hash matches.
    Returns (isValid, sha256_hash).
    """
    path = Path(path)
    if not path.exists():
        return False, ""
    try:
        file_hash = calculate_sha256(path)
        with np.load(path, allow_pickle=False) as data:
            if not set(REQUIRED_ARRAYS).issubset(data.files) or "metadata_json" not in data.files:
                return False, ""

            for name in REQUIRED_ARRAYS:
                arr = data[name]
                if not np.all(np.isfinite(arr)):
                    return False, ""

            # Check metadata
            meta_json = data["metadata_json"]
            meta_str = str(meta_json) if meta_json.ndim == 0 else str(meta_json[0])
            metadata = json.loads(meta_str)

            required_keys = {
                "dataset", "n_train", "fold_method", "source_config_hash",
                "contains_test_values", "contains_test_labels", "contains_evaluation_metrics"
            }
            if not required_keys.issubset(metadata.keys()):
                return False, ""

            if metadata["source_config_hash"] != expected_config_hash:
                return False, ""

            if expected_dataset is not None and metadata["dataset"] != expected_dataset:
                return False, ""

            for key in ("contains_test_values", "contains_test_labels", "contains_evaluation_metrics"):
                if metadata.get(key) is not False:
                    return False, ""

            if _has_forbidden_metadata(metadata):
                return False, ""

            n_train = int(metadata["n_train"])
            if n_train < 5:
                return False, ""

            # Verify shapes
            if len(data["fold_indices"]) != n_train:
                return False, ""

            if not np.array_equal(data["pseudo_run_indices"], np.arange(n_train)):
                return False, ""
            if np.any(data["fold_indices"] < 0):
                return False, ""
            if np.any(data["fit_counts"] <= 0) or np.any(data["fit_counts"] >= n_train):
                return False, ""
            if np.any(data["target_lengths"] < 8) or np.any(data["target_lengths"] > 2048):
                return False, ""

            for name in ("query_score_spectral", "query_score_shapelet", "query_score_interval"):
                if len(data[name]) != n_train:
                    return False, ""

            for name in ("thresholds_spectral", "thresholds_shapelet", "thresholds_interval",
                         "flags_spectral", "flags_shapelet", "flags_interval", "agreement_2of3", "agreement_3of3"):
                if data[name].shape != (n_train, 4):
                    return False, ""

            source_flags = []
            for source in ("spectral", "shapelet", "interval"):
                scores = data[f"query_score_{source}"][:, None]
                thresholds = data[f"thresholds_{source}"]
                flags = data[f"flags_{source}"]
                if flags.dtype != np.bool_:
                    return False, ""
                expected = np.isfinite(scores) & (scores > thresholds)
                if not np.array_equal(flags, expected):
                    return False, ""
                source_flags.append(flags)

            source_count = np.stack(source_flags, axis=0).sum(axis=0)
            if not np.array_equal(data["agreement_2of3"], source_count >= 2):
                return False, ""
            if not np.array_equal(data["agreement_3of3"], source_count >= 3):
                return False, ""

        return True, file_hash
    except Exception:
        return False, ""
