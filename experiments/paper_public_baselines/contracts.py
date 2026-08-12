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


@dataclass(frozen=True, slots=True)
class ScoreResult:
    method_id: str
    status: str
    reason: str | None
    oof_scores: np.ndarray
    test_scores: np.ndarray

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        oof = np.asarray(self.oof_scores, dtype=np.float64).reshape(-1).copy()
        test = np.asarray(self.test_scores, dtype=np.float64).reshape(-1).copy()
        if self.status == "complete":
            if self.reason is not None or not len(oof) or not len(test):
                raise ValueError("complete score result is inconsistent")
            if not np.isfinite(oof).all() or not np.isfinite(test).all():
                raise ValueError("complete scores must be finite")
        elif self.status == "abstain":
            if not self.reason or len(oof) or len(test):
                raise ValueError("abstaining score result is inconsistent")
        else:
            raise ValueError("score status must be complete or abstain")
        oof.setflags(write=False)
        test.setflags(write=False)
        object.__setattr__(self, "oof_scores", oof)
        object.__setattr__(self, "test_scores", test)

    @property
    def fit_rows(self) -> int:
        """Number of normal TRAIN rows represented by the OOF calibration."""
        return len(self.oof_scores) if self.status == "complete" else 0


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
