"""Deterministic TRAIN-defined preprocessing for public baselines."""

from __future__ import annotations

import numpy as np

from .contracts import DatasetValues, PreparedDataset


def prepare_dataset(values: DatasetValues) -> PreparedDataset:
    """Repair, TRAIN-align, and per-series normalize one dataset."""
    target_length = _train_target_length(values.train_rows)
    train = np.vstack([_align_row(row, target_length) for row in values.train_rows])
    test = np.vstack([_align_row(row, target_length) for row in values.test_rows])
    train = _z_normalize_rows(train)
    test = _z_normalize_rows(test)
    train.setflags(write=False)
    test.setflags(write=False)
    return PreparedDataset(
        dataset=values.dataset,
        train_indices=values.train_indices,
        train=train,
        test_indices=values.test_indices,
        test=test,
        target_length=target_length,
    )


def sanitize_series(row: np.ndarray) -> np.ndarray:
    """Interpolate partial non-finite runs and zero an entirely invalid row."""
    series = np.asarray(row, dtype=np.float32).reshape(-1)
    if not len(series):
        raise ValueError("series must not be empty")
    finite = np.isfinite(series)
    if finite.all():
        return series.copy()
    if not finite.any():
        return np.zeros_like(series, dtype=np.float32)
    positions = np.arange(len(series), dtype=np.float64)
    cleaned = series.copy()
    cleaned[~finite] = np.interp(
        positions[~finite], positions[finite], series[finite]
    )
    return cleaned.astype(np.float32)


def _train_target_length(rows: tuple[np.ndarray, ...]) -> int:
    lengths = tuple(len(np.asarray(row).reshape(-1)) for row in rows)
    if not lengths or min(lengths) <= 0:
        raise ValueError("TRAIN rows must be non-empty")
    return int(np.clip(np.median(lengths), 8, 2048))


def _align_row(row: np.ndarray, target_length: int) -> np.ndarray:
    cleaned = sanitize_series(row)
    if len(cleaned) == target_length:
        return cleaned
    source = np.linspace(0.0, 1.0, len(cleaned), dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    return np.interp(target, source, cleaned).astype(np.float32)


def _z_normalize_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    return ((values - center) / np.where(scale > 1e-8, scale, 1.0)).astype(
        np.float32
    )

