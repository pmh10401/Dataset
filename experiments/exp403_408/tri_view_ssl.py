"""Exp406 CPU-canonical tri-view self-supervised Shadow scoring.

The module consumes the already normalized values produced by ``foundation``.
It never accepts labels or lane objects: TRAIN-only fitting and calibration are
kept separate from the small helper that removes frozen Hard candidates from a
review-only Shadow list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math

import numpy as np
import torch
from torch import nn

from .contracts import (
    ALPHAS,
    FrozenConfig,
    LearnedSourceScores,
    ShadowSourcePrediction,
    train_eligibility,
)
from .foundation import crossfit_plan


VIEW_WIDTH = 64
HIDDEN_WIDTH = 64
EMBEDDING_WIDTH = 32
MAX_EPOCHS = 20
PATIENCE = 10
BATCH_SIZE = 32
MPS_SCORE_TOLERANCE = 1e-5


class TriViewEncoder(nn.Module):
    """Fixed-width encoders for the time, frequency, and residual views."""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(VIEW_WIDTH, HIDDEN_WIDTH),
                nn.GELU(),
                nn.Linear(HIDDEN_WIDTH, EMBEDDING_WIDTH),
            )
            for name in ("time", "frequency", "residual")
        })
        self.fused = nn.Linear(3 * EMBEDDING_WIDTH, EMBEDDING_WIDTH)
        self.decoders = nn.ModuleDict({
            name: nn.Linear(EMBEDDING_WIDTH, VIEW_WIDTH)
            for name in ("time", "frequency", "residual")
        })

    def forward(self, views: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        embeddings = {name: self.encoders[name](views[name]) for name in self.encoders}
        fused = self.fused(torch.cat([embeddings[name] for name in self.encoders], dim=1))
        reconstructions = {name: self.decoders[name](embeddings[name]) for name in self.decoders}
        return embeddings, fused, reconstructions


def build_tri_views(
    runs: object, *, prototype: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Build finite, fixed-width values-only views from normalized runs.

    ``prototype`` is deliberately explicit so TEST residuals can only be
    measured against a TRAIN-derived normal prototype.
    """
    values = _runs_matrix(runs, name="runs")
    baseline = (
        np.median(values, axis=0)
        if prototype is None
        else _prototype_vector(prototype, expected_length=values.shape[1])
    )
    time = _resample_rows(values, VIEW_WIDTH)
    spectrum = np.log1p(np.abs(np.fft.rfft(values, axis=1)))
    frequency = np.concatenate(
        (_resample_rows(spectrum, 48), _wavelet_band_energies(values, 16)), axis=1
    )
    residual_values = values - baseline.reshape(1, -1)
    residual = np.concatenate(
        (_resample_rows(residual_values, 48), _segment_means(np.abs(residual_values), 16)),
        axis=1,
    )
    result = {"time": time, "frequency": frequency, "residual": residual}
    if any(value.shape != (len(values), VIEW_WIDTH) or not np.isfinite(value).all() for value in result.values()):
        raise ValueError("tri-view construction produced non-finite fixed-width values")
    return result


def score_tri_view(
    train: object,
    test: object,
    config: FrozenConfig,
    device: str = "cpu",
) -> LearnedSourceScores:
    """Fit and calibrate Exp406 with TRAIN-only OOF evidence.

    CPU is canonical.  MPS is returned only after the same input and seed meet
    the frozen score-tolerance and candidate-rank parity checks against CPU.
    """
    _validate_config(config)
    train_values = _runs_matrix(train, name="train")
    test_values = _runs_matrix(test, name="test")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("TRAIN and TEST run lengths must match")
    if train_values.shape[1] != int(config.target_length):
        raise ValueError("run length must match FrozenConfig.target_length")
    if not train_eligibility(len(train_values)).learned_methods:
        return LearnedSourceScores(abstention_reason="n_train_below_50")

    requested = str(device).lower()
    if requested == "cpu":
        return _score_on_device(train_values, test_values, config, "cpu")
    if requested != "mps":
        raise ValueError("device must be cpu or mps")
    if not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    cpu = _score_on_device(train_values, test_values, config, "cpu")
    mps = _score_on_device(train_values, test_values, config, "mps")
    _require_mps_parity(cpu, mps)
    return mps


def shadow_predictions(
    dataset: str,
    scores: LearnedSourceScores,
    frozen_hard: Sequence[int] | frozenset[int],
) -> dict[str, tuple[ShadowSourcePrediction, ...]]:
    """Return review-only Shadow records outside the immutable Hard set.

    Standard and Priority lanes are intentionally absent from this API.  This
    helper does not mutate any frozen lane. Every returned record explicitly
    carries the ``self_supervised_shadow`` source marker.
    """
    hard = frozenset(int(index) for index in frozen_hard)
    if not scores.available:
        return {str(alpha): () for alpha in ALPHAS}
    p_values_by_alpha = {
        str(alpha): np.asarray(scores.scores_by_alpha.get(str(alpha), ()), dtype=float)
        for alpha in ALPHAS
    }
    expected_count = len(scores.raw_test_scores)
    if expected_count == 0 or not np.isfinite(np.asarray(scores.raw_test_scores, dtype=float)).all():
        raise ValueError("learned raw test scores are invalid")
    if any(values.ndim != 1 or len(values) != expected_count or not np.isfinite(values).all() for values in p_values_by_alpha.values()):
        raise ValueError("learned score alpha coverage is invalid")
    result: dict[str, tuple[ShadowSourcePrediction, ...]] = {}
    for alpha in ALPHAS:
        key = str(alpha)
        values = p_values_by_alpha[key]
        result[key] = tuple(
            ShadowSourcePrediction(
                dataset=str(dataset),
                candidate_index=int(index),
                scores_by_alpha={
                    other_key: float(other_values[index])
                    for other_key, other_values in p_values_by_alpha.items()
                },
                source="self_supervised_shadow",
            )
            for index, value in enumerate(values)
            if index not in hard and float(value) <= float(alpha)
        )
    return result


def _score_on_device(
    train: np.ndarray, test: np.ndarray, config: FrozenConfig, device: str
) -> LearnedSourceScores:
    prototype = np.median(train, axis=0)
    oof_scores = np.empty(len(train), dtype=float)
    for fold_number, fold in enumerate(crossfit_plan(len(train), seed=int(config.seed))):
        fit_indices = np.asarray(fold.train_indices, dtype=int)
        query_indices = np.asarray(fold.validation_indices, dtype=int)
        fit_views = build_tri_views(train[fit_indices])
        query_views = build_tri_views(train[query_indices], prototype=np.median(train[fit_indices], axis=0))
        model = _fit_encoder(fit_views, seed=int(config.seed) + fold_number, device=device)
        oof_scores[query_indices] = _anomaly_scores(model, fit_views, query_views, device=device)
    if not np.isfinite(oof_scores).all():
        raise ValueError("non-finite TRAIN OOF self-supervised scores")

    train_views = build_tri_views(train, prototype=prototype)
    test_views = build_tri_views(test, prototype=prototype)
    final_model = _fit_encoder(train_views, seed=int(config.seed), device=device)
    test_scores = _anomaly_scores(final_model, train_views, test_views, device=device)
    p_values = _empirical_p_values(oof_scores, test_scores)
    return LearnedSourceScores(
        scores_by_alpha={str(alpha): tuple(float(value) for value in p_values) for alpha in ALPHAS},
        raw_test_scores=tuple(float(value) for value in test_scores),
        available=True,
    )


def _fit_encoder(views: Mapping[str, np.ndarray], *, seed: int, device: str) -> TriViewEncoder:
    count = len(next(iter(views.values())))
    fit_indices, validation_indices = _train_validation_split(count, seed=seed)
    model = _new_model(seed=seed, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit_views = _tensor_views(_select_rows(views, fit_indices), device=device)
    validation_views = _tensor_views(_select_rows(views, validation_indices), device=device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    stale_epochs = 0
    model.train()
    for epoch in range(MAX_EPOCHS):
        for start in range(0, len(fit_indices), BATCH_SIZE):
            batch = {name: value[start : start + BATCH_SIZE] for name, value in fit_views.items()}
            optimizer.zero_grad(set_to_none=True)
            _training_loss(model, batch, phase=epoch).backward()
            optimizer.step()
        validation_loss = _validation_loss(model, validation_views)
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("TRAIN-only validation did not produce an encoder checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model


def _training_loss(model: TriViewEncoder, views: Mapping[str, torch.Tensor], *, phase: int) -> torch.Tensor:
    masked, masks = _masked_views(views, phase=phase)
    embeddings, fused, reconstructions = model(masked)
    reconstruction = torch.stack([
        ((reconstructions[name] - views[name])[:, masks[name]]) ** 2
        for name in embeddings
    ]).mean()
    consistency = (
        ((embeddings["time"] - embeddings["frequency"]) ** 2).mean()
        + ((embeddings["time"] - embeddings["residual"]) ** 2).mean()
        + ((embeddings["frequency"] - embeddings["residual"]) ** 2).mean()
    ) / 3.0
    prototype = fused.detach().mean(dim=0, keepdim=True)
    compactness = ((fused - prototype) ** 2).mean()
    return reconstruction + consistency + 0.1 * compactness


def _validation_loss(model: TriViewEncoder, views: Mapping[str, torch.Tensor]) -> float:
    model.eval()
    with torch.no_grad():
        loss = _training_loss(model, views, phase=0)
    model.train()
    return float(loss.detach().cpu())


def _anomaly_scores(
    model: TriViewEncoder,
    reference_views: Mapping[str, np.ndarray],
    query_views: Mapping[str, np.ndarray],
    *,
    device: str,
) -> np.ndarray:
    reference = _tensor_views(reference_views, device=device)
    query = _tensor_views(query_views, device=device)
    model.eval()
    with torch.no_grad():
        _, reference_fused, _ = model(reference)
        _, query_fused, reconstructions = model(query)
        center = reference_fused.mean(dim=0, keepdim=True)
        compactness = ((query_fused - center) ** 2).mean(dim=1)
        reconstruction = torch.stack([
            ((reconstructions[name] - query[name]) ** 2).mean(dim=1)
            for name in reconstructions
        ]).mean(dim=0)
        scores = compactness + reconstruction
    result = scores.detach().cpu().numpy().astype(float, copy=False)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("non-finite self-supervised query scores")
    return result


def _new_model(*, seed: int, device: str) -> TriViewEncoder:
    torch.manual_seed(int(seed))
    if device == "mps":
        torch.mps.manual_seed(int(seed))
    return TriViewEncoder().to(torch.device(device))


def _masked_views(
    views: Mapping[str, torch.Tensor], *, phase: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    mask = (torch.arange(VIEW_WIDTH, device=next(iter(views.values())).device) + int(phase)) % 4 == 0
    masks = {name: mask for name in views}
    return ({name: value.masked_fill(mask, 0.0) for name, value in views.items()}, masks)


def _tensor_views(views: Mapping[str, np.ndarray], *, device: str) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)
        for name, value in views.items()
    }


def _select_rows(views: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(value)[indices] for name, value in views.items()}


def _train_validation_split(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 3:
        raise ValueError("self-supervised fitting requires at least three TRAIN rows")
    shuffled = np.arange(count, dtype=int)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    validation_count = max(1, int(math.ceil(count * 0.2)))
    validation = np.sort(shuffled[:validation_count])
    fit = np.sort(shuffled[validation_count:])
    if len(fit) < 2:
        raise ValueError("self-supervised fitting needs two TRAIN fit rows")
    return fit, validation


def _runs_matrix(values: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional run matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite normalized values")
    return matrix


def _prototype_vector(values: object, *, expected_length: int) -> np.ndarray:
    prototype = np.asarray(values, dtype=float).reshape(-1)
    if len(prototype) != expected_length or not np.isfinite(prototype).all():
        raise ValueError("prototype must be finite and match run length")
    return prototype


def _resample_rows(values: np.ndarray, width: int) -> np.ndarray:
    positions = np.linspace(0.0, values.shape[1] - 1.0, int(width))
    source = np.arange(values.shape[1], dtype=float)
    return np.asarray([np.interp(positions, source, row) for row in values], dtype=float)


def _segment_means(values: np.ndarray, segments: int) -> np.ndarray:
    return np.asarray([
        [float(np.mean(part)) for part in np.array_split(row, int(segments))]
        for row in values
    ], dtype=float)


def _wavelet_band_energies(values: np.ndarray, bands: int) -> np.ndarray:
    details = np.diff(values, axis=1, prepend=values[:, :1])
    return _segment_means(np.abs(details), bands)


def _empirical_p_values(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray([
        (1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0)
        for value in query
    ], dtype=float)


def _require_mps_parity(cpu: LearnedSourceScores, mps: LearnedSourceScores) -> None:
    if cpu.available != mps.available or cpu.abstention_reason != mps.abstention_reason:
        raise ValueError("MPS parity mismatch")
    if not cpu.available:
        return
    cpu_raw = np.asarray(cpu.raw_test_scores, dtype=float)
    mps_raw = np.asarray(mps.raw_test_scores, dtype=float)
    if cpu_raw.ndim != 1 or mps_raw.ndim != 1 or cpu_raw.shape != mps_raw.shape:
        raise ValueError("MPS raw score shape mismatch")
    if not np.isfinite(cpu_raw).all() or not np.isfinite(mps_raw).all():
        raise ValueError("MPS raw scores must be finite")
    if not np.array_equal(
        np.argsort(-cpu_raw, kind="stable"), np.argsort(-mps_raw, kind="stable")
    ):
        raise ValueError("MPS raw candidate rank mismatch")
    if float(np.max(np.abs(cpu_raw - mps_raw), initial=0.0)) > MPS_SCORE_TOLERANCE:
        raise ValueError("MPS raw score tolerance mismatch")


def _validate_config(config: FrozenConfig) -> None:
    if not isinstance(config, FrozenConfig):
        raise TypeError("config must be FrozenConfig")
    if int(config.seed) != config.seed:
        raise ValueError("FrozenConfig.seed must be an integer")
    if int(config.target_length) < 2:
        raise ValueError("FrozenConfig.target_length must be at least two")
    if tuple(config.alphas) != ALPHAS:
        raise ValueError("FrozenConfig.alphas must match the frozen alpha contract")
