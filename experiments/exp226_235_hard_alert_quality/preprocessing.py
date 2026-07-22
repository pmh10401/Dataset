from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import DatasetBundle


PREPROCESSING_ORDER = (
    "missing_repair",
    "train_length_alignment",
    "normalization",
)


def preprocessing_contract() -> dict[str, object]:
    return {
        "order": list(PREPROCESSING_ORDER),
        "missing_repair_grid": "original_series",
        "target_length_source": "train_normal_only",
        "normalization_after_alignment": True,
        "test_wide_normalization": False,
    }


def sanitize_series(values: np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(series) == 0:
        raise ValueError("series must not be empty")
    finite = np.isfinite(series)
    if finite.all():
        return series
    if not finite.any():
        return np.zeros_like(series, dtype=np.float32)
    positions = np.arange(len(series))
    cleaned = series.copy()
    cleaned[~finite] = np.interp(
        positions[~finite], positions[finite], series[finite]
    )
    return cleaned.astype(np.float32)


def train_target_length(train_rows: Sequence[np.ndarray]) -> int:
    lengths = [len(np.asarray(row).reshape(-1)) for row in train_rows]
    if not lengths or min(lengths) <= 0:
        raise ValueError("TRAIN rows must be non-empty")
    return int(np.clip(np.median(lengths), 8, 2048))


def _align_row(values: np.ndarray, target_length: int) -> np.ndarray:
    row = sanitize_series(values)
    if len(row) == target_length:
        return row.astype(np.float32, copy=False)
    source = np.linspace(0.0, 1.0, len(row), dtype=np.float64)
    target = np.linspace(0.0, 1.0, target_length, dtype=np.float64)
    return np.interp(target, source, row).astype(np.float32)


def align_bundle(bundle: DatasetBundle) -> DatasetBundle:
    target = train_target_length(bundle.train_normal)
    train = np.vstack([_align_row(row, target) for row in bundle.train_normal])
    test = np.vstack([_align_row(row, target) for row in bundle.test_values])
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError(f"non-finite aligned values: dataset={bundle.dataset}")
    return DatasetBundle(
        dataset=bundle.dataset,
        train_normal=train,
        test_values=test,
        target_length=target,
    )
