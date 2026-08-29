"""Immutable Task 1 contracts for Exp479."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config_antigravity import (
    EXCLUDED_DATASETS_ANTIGRAVITY,
    EXPECTED_EXISTING_SCORE_DATASETS_ANTIGRAVITY,
    EXPECTED_INCLUDED_DATASETS_ANTIGRAVITY,
    EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY,
    EXPECTED_SCORE_CHECKPOINTS_ANTIGRAVITY,
)


@dataclass(frozen=True, slots=True)
class PopulationAuditAntigravity:
    """Exact set-difference evidence used by every later Exp479 task."""

    datasets: tuple[str, ...]
    excluded_datasets: tuple[str, ...]
    included_count: int
    existing_score_count: int
    checkpoint_count: int

    def __post_init__(self) -> None:
        if type(self.datasets) is not tuple or any(
            type(dataset) is not str or not dataset for dataset in self.datasets
        ):
            raise ValueError("datasets must be an exact tuple of non-empty strings")
        if len(self.datasets) != EXPECTED_RECOVERY_DATASETS_ANTIGRAVITY:
            raise ValueError("Exp479 requires exactly 155 recovery datasets")
        if self.datasets != tuple(sorted(self.datasets)):
            raise ValueError("recovery datasets must be sorted")
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("recovery datasets must be unique")
        if self.excluded_datasets != EXCLUDED_DATASETS_ANTIGRAVITY:
            raise ValueError("Exp479 exclusion contract mismatch")
        exact_counts = (
            (self.included_count, EXPECTED_INCLUDED_DATASETS_ANTIGRAVITY),
            (self.existing_score_count, EXPECTED_EXISTING_SCORE_DATASETS_ANTIGRAVITY),
            (self.checkpoint_count, EXPECTED_SCORE_CHECKPOINTS_ANTIGRAVITY),
        )
        if any(type(actual) is not int or actual != expected for actual, expected in exact_counts):
            raise ValueError("Exp479 population counts do not match the frozen contract")


@dataclass(frozen=True, slots=True)
class PreparedDatasetAntigravity:
    """Finite equal-length value arrays prepared without reading TEST labels."""

    dataset: str
    train_rows: np.ndarray
    test_rows: np.ndarray
    test_indices: tuple[int, ...]
    target_length: int
    raw_train_rows: int
    excluded_train_rows: int
    raw_test_rows: int
    excluded_test_rows: int

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("dataset must be an exact non-empty string")
        if type(self.target_length) is not int or self.target_length < 1:
            raise ValueError("target_length must be a positive exact integer")
        for field in (
            "raw_train_rows",
            "excluded_train_rows",
            "raw_test_rows",
            "excluded_test_rows",
        ):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field} must be a non-negative exact integer")
        if self.raw_train_rows - self.excluded_train_rows < 1:
            raise ValueError("at least one finite TRAIN normal row is required")
        if self.raw_test_rows - self.excluded_test_rows < 1:
            raise ValueError("at least one finite TEST row is required")
        if type(self.test_indices) is not tuple or any(
            type(index) is not int or index < 0 for index in self.test_indices
        ):
            raise ValueError("test_indices must be an exact tuple of non-negative integers")
        if len(set(self.test_indices)) != len(self.test_indices):
            raise ValueError("test_indices must be unique")
        for field in ("train_rows", "test_rows"):
            values = getattr(self, field)
            if (
                type(values) is not np.ndarray
                or values.dtype != np.float64
                or values.ndim != 2
                or values.shape[1] != self.target_length
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"{field} must be a finite float64 matrix at target_length")
            protected = np.frombuffer(values.tobytes(order="C"), dtype=np.float64).reshape(
                values.shape
            )
            object.__setattr__(self, field, protected)
        if len(self.test_indices) != len(self.test_rows):
            raise ValueError("test_indices must align with retained TEST rows")


__all__ = ["PopulationAuditAntigravity", "PreparedDatasetAntigravity"]
