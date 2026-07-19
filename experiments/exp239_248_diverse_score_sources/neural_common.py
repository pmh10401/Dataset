from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn

from experiments.exp239_248_diverse_score_sources.source_base import resample_rows


MAX_NEURAL_LENGTH = 256


def set_deterministic(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_rows(rows: np.ndarray) -> np.ndarray:
    return resample_rows(np.asarray(rows, dtype=np.float32), MAX_NEURAL_LENGTH).astype(np.float32)


def tensor_rows(rows: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(rows[:, None, :], dtype=torch.float32, device=torch.device(device))


class CompactEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.pool(self.network(values)).squeeze(-1)


class MaskedAutoencoder(nn.Module):
    def __init__(self, length: int) -> None:
        super().__init__()
        self.length = int(length)
        self.encoder = CompactEncoder()
        base = max(1, int(np.ceil(length / 8)))
        self.base = base
        self.expand = nn.Linear(32, 32 * base)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        hidden = self.expand(embedding).reshape(len(values), 32, self.base)
        reconstruction = self.decoder(hidden)
        if reconstruction.shape[-1] != self.length:
            reconstruction = nn.functional.interpolate(
                reconstruction, size=self.length, mode="linear", align_corners=False
            )
        return reconstruction, embedding

