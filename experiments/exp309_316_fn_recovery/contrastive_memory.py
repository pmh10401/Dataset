from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch import nn
from torch.nn import functional as F

from .config import (
    NEURAL_BATCH_SIZE, NEURAL_EPOCHS, NEURAL_LEARNING_RATE,
    NEURAL_MIN_LENGTH, NEURAL_MIN_TRAIN, NEURAL_PATIENCE,
)
from .contracts import ScoreArtifact
from .neural_common import (
    choose_device, clear_device_cache, early_stop_train, seed_everything,
    state_dict_sha256, train_validation_calibration_split,
)


class TemporalEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2, dilation=1), nn.ReLU(),
            nn.Conv1d(16, 24, 5, padding=4, dilation=2), nn.ReLU(),
            nn.Conv1d(24, 32, 5, padding=8, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(32, int(embedding_dim), bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.network(values).squeeze(-1)), dim=1)


def _augment(values: torch.Tensor) -> torch.Tensor:
    output = values * (1.0 + 0.05 * torch.randn((len(values), 1, 1), device=values.device))
    output = output + 0.02 * torch.randn_like(output)
    length = output.shape[-1]; mask_length = max(1, length // 10)
    starts = torch.randint(0, max(1, length - mask_length + 1), (len(values),), device=values.device)
    positions = torch.arange(length, device=values.device)[None, :]
    mask = (positions >= starts[:, None]) & (positions < starts[:, None] + mask_length)
    return output.masked_fill(mask[:, None, :], 0.0)


def _contrastive_loss(model: nn.Module, values: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    first, second = model(_augment(values)), model(_augment(values))
    logits = first @ second.T / float(temperature)
    labels = torch.arange(len(values), device=values.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _embed(model: nn.Module, rows: np.ndarray, device: torch.device) -> np.ndarray:
    values = torch.from_numpy(np.asarray(rows, dtype=np.float32)[:, None, :]).to(device)
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), NEURAL_BATCH_SIZE):
            output.append(model(values[start:start + NEURAL_BATCH_SIZE]).detach().cpu().numpy())
    return np.vstack(output).astype(np.float32)


def _memory_distance(memory: np.ndarray, query: np.ndarray) -> np.ndarray:
    return cdist(np.asarray(query), np.asarray(memory), metric="euclidean").min(axis=1).astype(np.float64)


def score_contrastive_memory(
    train_normal: np.ndarray, test_values: np.ndarray, device: str | torch.device,
    seed: int, *, epochs: int = NEURAL_EPOCHS,
) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32); test = np.asarray(test_values, dtype=np.float32)
    if len(train) < NEURAL_MIN_TRAIN or train.shape[1] < NEURAL_MIN_LENGTH:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"model": "temporal_contrastive_encoder"}, False, "neural_eligibility")
    seed_everything(seed)
    target_device = choose_device(device) if isinstance(device, str) else device
    fit_indices, validation_indices, calibration_indices = train_validation_calibration_split(len(train), seed)
    fit = torch.from_numpy(train[fit_indices, None, :]).to(target_device)
    validation = torch.from_numpy(train[validation_indices, None, :]).to(target_device)
    model = TemporalEncoder(32).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_LEARNING_RATE)

    def train_step():
        losses = []
        order = torch.randperm(len(fit), device=target_device)
        for start in range(0, len(fit), NEURAL_BATCH_SIZE):
            batch = fit[order[start:start + NEURAL_BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True); loss = _contrastive_loss(model, batch); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        return np.mean(losses)

    def validation_step():
        return float(_contrastive_loss(model, validation).detach().cpu())

    history = early_stop_train(model, optimizer, train_step, validation_step, epochs=epochs, patience=min(NEURAL_PATIENCE, max(1, epochs)))
    memory = _embed(model, train[fit_indices], target_device)
    calibration_scores = _memory_distance(memory, _embed(model, train[calibration_indices], target_device))
    test_scores = _memory_distance(memory, _embed(model, test, target_device))
    diagnostics = {
        "model": "temporal_contrastive_encoder", "embedding_dim": 32,
        "temperature": 0.1, "device": target_device.type,
        "fit_count": len(fit_indices), "validation_count": len(validation_indices),
        "calibration_count": len(calibration_indices), "training": history,
        "model_sha256": state_dict_sha256(model),
        "calibration_method": "three_way_train_split",
    }
    model.to("cpu"); clear_device_cache(target_device)
    return ScoreArtifact(calibration_scores, test_scores, diagnostics)
