from __future__ import annotations

import numpy as np
import torch

from .config import (
    NEURAL_BATCH_SIZE, NEURAL_LEARNING_RATE, NEURAL_MIN_LENGTH,
    NEURAL_MIN_TRAIN, NEURAL_PATIENCE,
)
from .contracts import ScoreArtifact
from .contrastive_memory import TemporalEncoder, _embed
from .neural_common import (
    choose_device, clear_device_cache, early_stop_train, seed_everything,
    state_dict_sha256, train_validation_calibration_split,
)


def _distance(embeddings: np.ndarray, center: np.ndarray) -> np.ndarray:
    return np.sum((np.asarray(embeddings, dtype=np.float64) - np.asarray(center, dtype=np.float64)) ** 2, axis=1)


def score_deep_svdd(
    train_normal: np.ndarray, test_values: np.ndarray, device: str | torch.device,
    seed: int, *, epochs: int = 10,
) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32); test = np.asarray(test_values, dtype=np.float32)
    if len(train) < NEURAL_MIN_TRAIN or train.shape[1] < NEURAL_MIN_LENGTH:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"model": "deep_svdd_encoder"}, False, "neural_eligibility")
    seed_everything(seed)
    target_device = choose_device(device) if isinstance(device, str) else device
    fit_indices, validation_indices, calibration_indices = train_validation_calibration_split(len(train), seed)
    fit = torch.from_numpy(train[fit_indices, None, :]).to(target_device)
    validation = torch.from_numpy(train[validation_indices, None, :]).to(target_device)
    model = TemporalEncoder(32).to(target_device)
    with torch.no_grad():
        center = model(fit).mean(dim=0).detach()
        center = torch.where(center.abs() < 1e-3, torch.sign(center + 1e-12) * 1e-3, center)
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_LEARNING_RATE)

    def loss_for(values: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum((model(values) - center) ** 2, dim=1))

    def train_step():
        losses = []
        order = torch.randperm(len(fit), device=target_device)
        for start in range(0, len(fit), NEURAL_BATCH_SIZE):
            batch = fit[order[start:start + NEURAL_BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True); loss = loss_for(batch); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        return np.mean(losses)

    def validation_step():
        return float(loss_for(validation).detach().cpu())

    history = early_stop_train(model, optimizer, train_step, validation_step, epochs=epochs, patience=min(NEURAL_PATIENCE, max(1, epochs)))
    center_np = center.detach().cpu().numpy()
    calibration_scores = _distance(_embed(model, train[calibration_indices], target_device), center_np)
    test_scores = _distance(_embed(model, test, target_device), center_np)
    diagnostics = {
        "model": "deep_svdd_encoder", "embedding_dim": 32,
        "device": target_device.type, "fit_count": len(fit_indices),
        "validation_count": len(validation_indices), "calibration_count": len(calibration_indices),
        "training": history, "model_sha256": state_dict_sha256(model),
        "center_sha256": __import__("hashlib").sha256(center_np.tobytes()).hexdigest(),
        "calibration_method": "three_way_train_split",
    }
    model.to("cpu"); clear_device_cache(target_device)
    return ScoreArtifact(calibration_scores, test_scores, diagnostics)
