from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import (
    NEURAL_BATCH_SIZE, NEURAL_EPOCHS, NEURAL_LEARNING_RATE,
    NEURAL_MIN_LENGTH, NEURAL_MIN_TRAIN, NEURAL_PATIENCE,
)
from .contracts import ScoreArtifact
from .neural_common import (
    choose_device, clear_device_cache, early_stop_train, seed_everything,
    state_dict_sha256, train_validation_calibration_split,
)


class TinyPredictor(nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, 5, padding=2), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, int(horizon))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(values).squeeze(-1))


def _windows(rows: np.ndarray, context: int, horizon: int, cap: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, owners = [], [], []
    step = max(1, int(horizon))
    for owner, row in enumerate(np.asarray(rows, dtype=np.float32)):
        for start in range(0, len(row) - context - horizon + 1, step):
            xs.append(row[start:start + context]); ys.append(row[start + context:start + context + horizon]); owners.append(owner)
    if not xs:
        raise ValueError("target length cannot produce prediction windows")
    if len(xs) > int(cap):
        chosen = np.linspace(0, len(xs) - 1, int(cap), dtype=int)
        xs = [xs[index] for index in chosen]; ys = [ys[index] for index in chosen]; owners = [owners[index] for index in chosen]
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(owners, dtype=int)


def _tensor_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int):
    for start in range(0, len(x), int(batch_size)):
        yield x[start:start + batch_size], y[start:start + batch_size]


def _run_scores(model: nn.Module, rows: np.ndarray, context: int, horizon: int, device: torch.device) -> np.ndarray:
    x, y, owners = _windows(rows, context, horizon)
    values = torch.from_numpy(x[:, None, :]).to(device)
    target = torch.from_numpy(y).to(device)
    losses = []
    model.eval()
    with torch.no_grad():
        for xb, yb in _tensor_batches(values, target, NEURAL_BATCH_SIZE):
            losses.extend(torch.mean((model(xb) - yb) ** 2, dim=1).detach().cpu().numpy().tolist())
    losses = np.asarray(losses, dtype=np.float64)
    return np.asarray([np.quantile(losses[owners == owner], 0.95) for owner in range(len(rows))], dtype=np.float64)


def score_predictive_residual(
    train_normal: np.ndarray, test_values: np.ndarray, device: str | torch.device,
    seed: int, *, epochs: int = NEURAL_EPOCHS,
) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32); test = np.asarray(test_values, dtype=np.float32)
    if len(train) < NEURAL_MIN_TRAIN or train.shape[1] < NEURAL_MIN_LENGTH:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"model": "tiny_tcn_predictor"}, False, "neural_eligibility")
    seed_everything(seed)
    target_device = choose_device(device) if isinstance(device, str) else device
    fit_indices, validation_indices, calibration_indices = train_validation_calibration_split(len(train), seed)
    context = min(64, train.shape[1] // 2); horizon = max(4, context // 8)
    fit_x, fit_y, _ = _windows(train[fit_indices], context, horizon)
    val_x, val_y, _ = _windows(train[validation_indices], context, horizon)
    fit_x_t = torch.from_numpy(fit_x[:, None, :]).to(target_device); fit_y_t = torch.from_numpy(fit_y).to(target_device)
    val_x_t = torch.from_numpy(val_x[:, None, :]).to(target_device); val_y_t = torch.from_numpy(val_y).to(target_device)
    model = TinyPredictor(horizon).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NEURAL_LEARNING_RATE)
    loss_fn = nn.MSELoss()

    def train_step():
        losses = []
        for xb, yb in _tensor_batches(fit_x_t, fit_y_t, NEURAL_BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        return np.mean(losses)

    def validation_step():
        return float(loss_fn(model(val_x_t), val_y_t).detach().cpu())

    history = early_stop_train(model, optimizer, train_step, validation_step, epochs=epochs, patience=min(NEURAL_PATIENCE, max(1, epochs)))
    calibration_scores = _run_scores(model, train[calibration_indices], context, horizon, target_device)
    test_scores = _run_scores(model, test, context, horizon, target_device)
    diagnostics = {
        "model": "tiny_tcn_predictor", "device": target_device.type,
        "context": context, "horizon": horizon,
        "fit_count": len(fit_indices), "validation_count": len(validation_indices),
        "calibration_count": len(calibration_indices), "training": history,
        "model_sha256": state_dict_sha256(model),
        "calibration_method": "three_way_train_split",
    }
    model.to("cpu"); clear_device_cache(target_device)
    return ScoreArtifact(calibration_scores, test_scores, diagnostics)
