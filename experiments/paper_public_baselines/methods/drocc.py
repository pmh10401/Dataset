"""DROCC (Goyal et al., 2020) with deterministic annulus adversaries."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contracts import ScoreResult
from .deep_common import (
    Encoder1D,
    OFFICIAL_DEEP_EPOCHS,
    as_tensor,
    crossfit_deep_scores,
    finite_numpy,
    seed_cpu,
)


class _DroccNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder1D()
        self.head = nn.Linear(16, 1, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(values)).reshape(-1)


def score_drocc(
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
    *,
    epochs: int = OFFICIAL_DEEP_EPOCHS,
) -> ScoreResult:
    """Generate 5-fold TRAIN OOF and full-fit TEST DROCC scores."""
    return crossfit_deep_scores("drocc_1d", train, test, seed, epochs, _fit_and_score)


def _fit_and_score(
    fit: np.ndarray, query: np.ndarray, seed: int, epochs: int
) -> np.ndarray:
    generator = seed_cpu(seed)
    model = _DroccNetwork().cpu()
    fit_tensor = as_tensor(fit)
    query_tensor = as_tensor(query)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-6
    )
    batch_size = min(32, len(fit_tensor))
    warmup_epochs = min(10, max(1, epochs // 5))
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(fit_tensor), generator=generator)
        for start in range(0, len(fit_tensor), batch_size):
            batch = fit_tensor[order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            normal_loss = F.binary_cross_entropy_with_logits(
                model(batch), torch.ones(len(batch))
            )
            if epoch < warmup_epochs:
                loss = normal_loss
            else:
                adversarial = _annulus_adversaries(model, batch, generator)
                abnormal_loss = F.binary_cross_entropy_with_logits(
                    model(adversarial), torch.zeros(len(batch))
                )
                loss = normal_loss + abnormal_loss
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = -model(query_tensor)
    return finite_numpy(scores, len(query))


def _annulus_adversaries(
    model: nn.Module,
    batch: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    radius = 1.0
    gamma = 2.0
    step_size = 0.1
    perturbation = torch.randn(
        batch.shape, dtype=batch.dtype, device=batch.device, generator=generator
    )
    perturbation = _normalize_rows(perturbation)
    scales = radius + (gamma - 1.0) * torch.rand(
        (len(batch), 1, 1), generator=generator
    )
    adversarial = (batch + perturbation * scales).detach()
    for _ in range(20):
        adversarial.requires_grad_(True)
        hard_negative_loss = F.binary_cross_entropy_with_logits(
            model(adversarial), torch.zeros(len(batch))
        )
        gradient = torch.autograd.grad(hard_negative_loss, adversarial)[0]
        with torch.no_grad():
            perturbation = adversarial - batch + step_size * _normalize_rows(gradient)
            norms = _row_norms(perturbation)
            projected_norms = norms.clamp(min=radius, max=gamma * radius)
            perturbation = perturbation * projected_norms / norms.clamp_min(1e-12)
            adversarial = (batch + perturbation).detach()
    return adversarial


def _row_norms(values: torch.Tensor) -> torch.Tensor:
    return values.reshape(len(values), -1).norm(dim=1).reshape(-1, 1, 1)


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / _row_norms(values).clamp_min(1e-12)


__all__ = ["score_drocc"]
