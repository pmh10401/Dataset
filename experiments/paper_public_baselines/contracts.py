"""Immutable values-only contracts for public baseline scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetValues:
    dataset: str
    train_indices: tuple[int, ...]
    train_rows: tuple[np.ndarray, ...]
    test_indices: tuple[int, ...]
    test_rows: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("dataset must not be empty")
        _validate_split(self.train_indices, self.train_rows, "TRAIN")
        _validate_split(self.test_indices, self.test_rows, "TEST")


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    dataset: str
    train_indices: tuple[int, ...]
    train: np.ndarray
    test_indices: tuple[int, ...]
    test: np.ndarray
    target_length: int

    def __post_init__(self) -> None:
        train = np.asarray(self.train)
        test = np.asarray(self.test)
        if train.ndim != 2 or test.ndim != 2:
            raise ValueError("prepared arrays must be two-dimensional")
        if train.shape != (len(self.train_indices), self.target_length):
            raise ValueError("prepared TRAIN shape is invalid")
        if test.shape != (len(self.test_indices), self.target_length):
            raise ValueError("prepared TEST shape is invalid")
        if not np.isfinite(train).all() or not np.isfinite(test).all():
            raise ValueError("prepared arrays must be finite")


def _validate_split(
    indices: tuple[int, ...], rows: tuple[np.ndarray, ...], split: str
) -> None:
    if not rows or len(indices) != len(rows):
        raise ValueError(f"{split} rows and indices must be non-empty and aligned")
    if indices != tuple(range(len(indices))):
        raise ValueError(f"{split} indices must be contiguous from zero")
    for row in rows:
        values = np.asarray(row).reshape(-1)
        if not len(values):
            raise ValueError(f"{split} rows must not be empty")

