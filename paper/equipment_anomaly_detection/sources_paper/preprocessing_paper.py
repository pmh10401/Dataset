"""TRAIN-only alignment and metadata copied from historical score preprocessing.

Provenance: run_rank_ensemble_calibration.py SHA256
1191d05fabd85ec174b78eee537710e2748e1c89faca7ec824894831df3d6a9a
(resample_series, align_series_lengths, z_normalize); and
run_original_improvement_experiment.py SHA256
dd314580082a362a3e55b8c900c37ab84cf361716056ea0b61e4b20ed6e652c9
(target_len_for_record).
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from ..contracts_paper import RawDataset, SourceDatasetMetadata


PAPER_SEED = 20260717
PROVENANCE_PAPER = {
    "run_rank_ensemble_calibration.py": ("resample_series", "align_series_lengths", "z_normalize"),
    "run_original_improvement_experiment.py": ("target_len_for_record",),
}


def resample_series_paper(values: np.ndarray, target_len: int) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(-1)
    if target_len <= 0:
        raise ValueError("target_len must be positive")
    if len(row) == target_len:
        return row.astype(np.float32, copy=False)
    if not len(row):
        return np.zeros(target_len, dtype=np.float32)
    return np.interp(np.linspace(0.0, 1.0, target_len), np.linspace(0.0, 1.0, len(row)), row).astype(np.float32)


def align_series_lengths_paper(rows: tuple[np.ndarray, ...], target_len: int) -> np.ndarray:
    if not rows:
        return np.empty((0, int(target_len)), dtype=np.float32)
    return np.stack([resample_series_paper(row, target_len) for row in rows]).astype(np.float32)


def z_normalize_paper(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return (values - mean) / np.where(std == 0.0, 1.0, std)


def has_only_finite_values_paper(values: RawDataset) -> bool:
    return all(np.asarray(row).ndim == 1 and len(row) and np.all(np.isfinite(row))
               for row in (*values.train_rows, *values.test_rows))


def source_metadata_paper(values: RawDataset) -> SourceDatasetMetadata:
    if type(values.metadata_length) is not int or values.metadata_length <= 0:
        raise ValueError("metadata_length must be a positive integer")
    lengths = [len(np.asarray(row)) for row in values.train_rows]
    if not lengths or min(lengths) <= 0:
        raise ValueError("TRAIN rows must be nonempty")
    actual_median = int(round(float(np.median(lengths))))
    actual_max = int(max(lengths))
    train_target = min(max(8, actual_median), 2048)
    return SourceDatasetMetadata(
        dataset=values.dataset, dataset_name="__virtual_run_dataset__", family="__family_neutral__",
        metadata_length=values.metadata_length, actual_len_median=actual_median,
        actual_len_max=actual_max, train_target_len=train_target,
        test_actual_len_median=train_target,
    )


def metadata_record_paper(metadata: SourceDatasetMetadata) -> dict[str, object]:
    """Provide historical key names without exposing TEST-derived settings."""
    return asdict(metadata)
