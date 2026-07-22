from __future__ import annotations

import hashlib
import io
import random

import numpy as np
import torch


def choose_device(requested: str = "auto", *, mps_available: bool | None = None) -> torch.device:
    requested = str(requested)
    available = torch.backends.mps.is_available() if mps_available is None else bool(mps_available)
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "mps" or (requested == "auto" and available):
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def train_only_split(n_train: int, seed: int, calibration_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    if int(n_train) < 2:
        raise ValueError("at least two TRAIN rows are required")
    indices = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(indices)
    calibration_count = max(1, int(np.ceil(len(indices) * float(calibration_fraction))))
    calibration_count = min(calibration_count, len(indices) - 1)
    return sorted(indices[calibration_count:].tolist()), sorted(indices[:calibration_count].tolist())


def train_validation_calibration_split(n_train: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    if int(n_train) < 6:
        raise ValueError("at least six TRAIN rows are required for three-way split")
    indices = np.arange(int(n_train), dtype=int)
    np.random.default_rng(int(seed)).shuffle(indices)
    holdout = max(1, int(np.ceil(len(indices) * 0.15)))
    holdout = min(holdout, (len(indices) - 2) // 2)
    calibration = indices[:holdout]
    validation = indices[holdout: 2 * holdout]
    fit = indices[2 * holdout:]
    return sorted(fit.tolist()), sorted(validation.tolist()), sorted(calibration.tolist())


def state_dict_sha256(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def early_stop_train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    train_step, validation_step, *, epochs: int, patience: int,
) -> dict[str, float | int]:
    best_loss = float("inf")
    best_state = None
    waiting = 0
    epochs_run = 0
    for epoch in range(int(epochs)):
        model.train(); train_loss = float(train_step())
        model.eval()
        with torch.no_grad(): validation_loss = float(validation_step())
        epochs_run = epoch + 1
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= int(patience): break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"epochs_run": epochs_run, "best_validation_loss": best_loss, "last_train_loss": train_loss}
