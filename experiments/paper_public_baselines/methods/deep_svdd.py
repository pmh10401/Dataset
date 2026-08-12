"""Deep SVDD (Ruff et al., 2018) with TRAIN-only OOF calibration."""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import ScoreResult
from .deep_common import (
    Encoder1D,
    OFFICIAL_DEEP_EPOCHS,
    as_tensor,
    crossfit_deep_scores,
    finite_numpy,
    seed_cpu,
)


def score_deep_svdd(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    *,
    epochs: int = OFFICIAL_DEEP_EPOCHS,
) -> ScoreResult:
    """Generate 5-fold TRAIN OOF and full-fit TEST Deep SVDD scores."""
    return crossfit_deep_scores(
        "deep_svdd_1d", train, test, seed, epochs, _fit_and_score
    )


def _fit_and_score(
    fit: np.ndarray, query: np.ndarray, seed: int, epochs: int
) -> np.ndarray:
    generator = seed_cpu(seed)
    model = Encoder1D().cpu()
    fit_tensor = as_tensor(fit)
    query_tensor = as_tensor(query)
    with torch.no_grad():
        center = model(fit_tensor).mean(dim=0).detach()
        signs = torch.where(center >= 0, 1.0, -1.0)
        center = torch.where(center.abs() < 0.1, signs * 0.1, center)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-6
    )
    batch_size = min(32, len(fit_tensor))
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(fit_tensor), generator=generator)
        for start in range(0, len(fit_tensor), batch_size):
            batch = fit_tensor[order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            distance = torch.sum((model(batch) - center) ** 2, dim=1)
            loss = distance.mean()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sum((model(query_tensor) - center) ** 2, dim=1)
    return finite_numpy(scores, len(query))


__all__ = ["score_deep_svdd"]
