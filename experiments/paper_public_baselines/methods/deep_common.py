"""Deterministic CPU helpers shared by published deep one-class methods."""

from __future__ import annotations

from collections.abc import Callable
import random

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch import nn

from ..contracts import ScoreResult


MIN_DEEP_TRAIN_ROWS = 8
OFFICIAL_DEEP_EPOCHS = 50


class Encoder1D(nn.Module):
    """Frozen bias-free 1D representation used by both deep baselines."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, 5, padding=2, bias=False),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 16, bias=False),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


FoldScorer = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]


def crossfit_deep_scores(
    method_id: str,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    epochs: int,
    fold_scorer: FoldScorer,
) -> ScoreResult:
    """Return 5-fold TRAIN OOF scores and a separately full-fit TEST score."""
    train_values, test_values = validate_deep_matrices(train, test)
    if len(train_values) < MIN_DEEP_TRAIN_ROWS:
        return ScoreResult(
            method_id=method_id,
            status="abstain",
            reason="insufficient_train_rows",
            oof_scores=np.empty(0),
            test_scores=np.empty(0),
        )
    if type(epochs) is not int or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    oof = np.empty(len(train_values), dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    for fold, (fit_indices, held_indices) in enumerate(splitter.split(train_values)):
        oof[held_indices] = fold_scorer(
            train_values[fit_indices],
            train_values[held_indices],
            int(seed) + fold,
            epochs,
        )
    test_scores = fold_scorer(
        train_values, test_values, int(seed) + 10_000, epochs
    )
    return ScoreResult(
        method_id=method_id,
        status="complete",
        reason=None,
        oof_scores=oof,
        test_scores=test_scores,
    )


def validate_deep_matrices(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    train_values = np.asarray(train, dtype=np.float32)
    test_values = np.asarray(test, dtype=np.float32)
    if train_values.ndim != 2 or test_values.ndim != 2:
        raise ValueError("TRAIN and TEST must be two-dimensional")
    if not len(train_values) or not len(test_values):
        raise ValueError("TRAIN and TEST must not be empty")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("TRAIN and TEST lengths must match")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise ValueError("TRAIN and TEST must be finite")
    return train_values, test_values


def seed_cpu(seed: int) -> torch.Generator:
    """Freeze all random streams and require deterministic CPU kernels."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def as_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32)[:, None, :].copy())


def finite_numpy(values: torch.Tensor, expected_rows: int) -> np.ndarray:
    result = values.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if result.shape != (expected_rows,) or not np.isfinite(result).all():
        raise ValueError("deep method produced invalid scores")
    return result
