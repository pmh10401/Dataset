"""Exp407 TRAIN-synthetic anomaly Shadow evidence.

This source consumes the normalized values emitted by ``foundation``.  It
generates deterministic synthetic anomalies from normal TRAIN runs only,
calibrates each classifier against TRAIN OOF normal scores, and never accepts
TEST labels or mutable lane objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .contracts import (
    ALPHAS,
    FrozenConfig,
    LearnedSourceScores,
    SEED,
    ShadowSourcePrediction,
    train_eligibility,
)
from .foundation import crossfit_plan


SYNTHETIC_PERTURBATION_TYPES = (
    "spike",
    "drift",
    "level_shift",
    "flat_line",
    "amplitude_scale",
    "frequency_warp",
    "phase_order",
)
MODEL_NAMES = (
    "logistic_regression",
    "hist_gradient_boosting",
    "small_mlp",
)
FEATURE_WIDTH = 64
MAD_SCALE_FACTOR = 1.4826
# Zero or near-zero MAD gets this frozen floor; variance is never consulted.
MAD_DEGENERATE_SCALE = 1e-8


@dataclass(frozen=True)
class SyntheticAnomalyConfig:
    """All Exp407 generation and model choices frozen before TEST scoring."""

    frozen: FrozenConfig = field(default_factory=FrozenConfig)
    examples_per_type_per_run: int = 1
    spike_mad_multiplier: float = 5.0
    drift_mad_multiplier: float = 4.0
    level_shift_mad_multiplier: float = 3.0
    flat_line_fraction: float = 0.25
    amplitude_scale: float = 1.5
    frequency_warp_factor: float = 1.25
    phase_order_segments: int = 4
    mlp_epochs: int = 25
    mlp_learning_rate: float = 1e-3


@dataclass(frozen=True)
class SyntheticAnomalies:
    """Synthetic values and their deterministic content hash."""

    values: np.ndarray
    by_type: Mapping[str, np.ndarray]
    sha256: str


class SmallMLP(nn.Module):
    """The frozen Exp407 64-hidden-unit binary classifier."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).reshape(-1)


def generate_synthetic_anomalies(
    train: object, config: SyntheticAnomalyConfig, seed: int
) -> SyntheticAnomalies:
    """Generate every frozen perturbation from TRAIN median, MAD, and length.

    Each perturbation type contributes exactly
    ``examples_per_type_per_run * n_train`` rows.  No TEST value participates
    in the magnitude, position, count, or ordering of generated examples.
    """
    frozen = _validate_config(config)
    values = _runs_matrix(train, name="train")
    if values.shape[1] != frozen.target_length:
        raise ValueError("run length must match FrozenConfig.target_length")
    rng = np.random.default_rng(int(seed))
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(MAD_SCALE_FACTOR * mad, MAD_DEGENERATE_SCALE)
    generated: dict[str, np.ndarray] = {}
    for name in SYNTHETIC_PERTURBATION_TYPES:
        rows = [
            _perturb(row, name, median=median, scale=scale, config=config, rng=rng)
            for row in values
            for _ in range(int(config.examples_per_type_per_run))
        ]
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (
            len(values) * int(config.examples_per_type_per_run), values.shape[1]
        ) or not np.isfinite(matrix).all():
            raise ValueError(f"synthetic {name} perturbation is invalid")
        matrix.setflags(write=False)
        generated[name] = matrix
    combined = np.vstack([generated[name] for name in SYNTHETIC_PERTURBATION_TYPES])
    combined.setflags(write=False)
    return SyntheticAnomalies(
        values=combined,
        by_type=MappingProxyType(generated),
        sha256=_synthetic_sha256(generated),
    )


def score_synthetic_models(
    train: object, test: object, config: SyntheticAnomalyConfig
) -> tuple[LearnedSourceScores, ...]:
    """Fit all frozen models on TRAIN evidence and score TEST without selection."""
    frozen = _validate_config(config)
    train_values = _runs_matrix(train, name="train")
    test_values = _runs_matrix(test, name="test")
    if train_values.shape[1] != test_values.shape[1]:
        raise ValueError("TRAIN and TEST run lengths must match")
    if train_values.shape[1] != frozen.target_length:
        raise ValueError("run length must match FrozenConfig.target_length")
    if not train_eligibility(len(train_values)).learned_methods:
        return tuple(
            LearnedSourceScores(
                abstention_reason="n_train_below_50", model_name=model_name
            )
            for model_name in MODEL_NAMES
        )

    test_features = _model_features(test_values)
    results: list[LearnedSourceScores] = []
    for model_index, model_name in enumerate(MODEL_NAMES):
        model_seed = int(frozen.seed) + model_index * 10_000
        oof_scores = _crossfit_normal_scores(
            train_values, config=config, model_name=model_name, seed=model_seed
        )
        final_model = _fit_model(
            train_values,
            config=config,
            model_name=model_name,
            seed=model_seed,
        )
        raw_test_scores = _predict_scores(final_model, test_features, model_name)
        if not np.isfinite(oof_scores).all() or not np.isfinite(raw_test_scores).all():
            raise ValueError(f"non-finite {model_name} scores")
        p_values = _empirical_p_values(oof_scores, raw_test_scores)
        results.append(
            LearnedSourceScores(
                scores_by_alpha={
                    str(alpha): tuple(float(value) for value in p_values)
                    for alpha in ALPHAS
                },
                raw_test_scores=tuple(float(value) for value in raw_test_scores),
                available=True,
                model_name=model_name,
            )
        )
    return tuple(results)


def synthetic_shadow_predictions(
    dataset: str,
    scores: LearnedSourceScores,
    frozen_hard: Sequence[int] | frozenset[int],
) -> dict[str, tuple[ShadowSourcePrediction, ...]]:
    """Return review-only synthetic-anomaly evidence outside immutable Hard."""
    hard = frozenset(int(index) for index in frozen_hard)
    if not scores.available:
        return {str(alpha): () for alpha in ALPHAS}
    raw = np.asarray(scores.raw_test_scores, dtype=float)
    p_values_by_alpha = {
        str(alpha): np.asarray(scores.scores_by_alpha.get(str(alpha), ()), dtype=float)
        for alpha in ALPHAS
    }
    if raw.ndim != 1 or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("synthetic raw test scores are invalid")
    if any(
        values.ndim != 1 or len(values) != len(raw) or not np.isfinite(values).all()
        for values in p_values_by_alpha.values()
    ):
        raise ValueError("synthetic score alpha coverage is invalid")
    return {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=str(dataset),
                candidate_index=int(index),
                scores_by_alpha={
                    key: float(values[index])
                    for key, values in p_values_by_alpha.items()
                },
                source="synthetic_anomaly_shadow",
            )
            for index, value in enumerate(p_values_by_alpha[str(alpha)])
            if index not in hard and float(value) <= float(alpha)
        )
        for alpha in ALPHAS
    }


def _crossfit_normal_scores(
    train: np.ndarray,
    *,
    config: SyntheticAnomalyConfig,
    model_name: str,
    seed: int,
) -> np.ndarray:
    scores = np.empty(len(train), dtype=float)
    for fold_number, fold in enumerate(crossfit_plan(len(train), seed=seed)):
        fit_indices = np.asarray(fold.train_indices, dtype=int)
        query_indices = np.asarray(fold.validation_indices, dtype=int)
        model = _fit_model(
            train[fit_indices],
            config=config,
            model_name=model_name,
            seed=seed + fold_number + 1,
        )
        scores[query_indices] = _predict_scores(
            model, _model_features(train[query_indices]), model_name
        )
    return scores


def _fit_model(
    normal_train: np.ndarray,
    *,
    config: SyntheticAnomalyConfig,
    model_name: str,
    seed: int,
) -> object:
    synthetic = generate_synthetic_anomalies(normal_train, config, seed).values
    normal_features = _model_features(normal_train)
    synthetic_features = _model_features(synthetic)
    features = np.vstack((normal_features, synthetic_features))
    labels = np.concatenate((np.zeros(len(normal_features), dtype=int), np.ones(len(synthetic_features), dtype=int)))
    if model_name == "logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=int(seed)
            ),
        )
        return model.fit(features, labels)
    if model_name == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=int(seed),
        )
        weights = np.where(labels == 0, 0.5 / len(normal_features), 0.5 / len(synthetic_features))
        return model.fit(features, labels, sample_weight=weights)
    if model_name == "small_mlp":
        return _fit_small_mlp(features, labels, config=config, seed=seed)
    raise ValueError(f"unknown synthetic classifier: {model_name}")


def _fit_small_mlp(
    features: np.ndarray, labels: np.ndarray, *, config: SyntheticAnomalyConfig, seed: int
) -> tuple[SmallMLP, np.ndarray, np.ndarray]:
    torch.manual_seed(int(seed))
    center = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    normalized = (features - center) / np.where(scale > 1e-8, scale, 1.0)
    model = SmallMLP(normalized.shape[1], hidden_dim=64, output_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.mlp_learning_rate))
    inputs = torch.as_tensor(normalized, dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.float32)
    positive_weight = torch.tensor(
        [float(np.count_nonzero(labels == 0)) / float(np.count_nonzero(labels == 1))],
        dtype=torch.float32,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    model.train()
    for _ in range(int(config.mlp_epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(inputs), targets).backward()
        optimizer.step()
    model.eval()
    return model, center, scale


def _predict_scores(model: object, features: np.ndarray, model_name: str) -> np.ndarray:
    if model_name in {"logistic_regression", "hist_gradient_boosting"}:
        scores = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    elif model_name == "small_mlp":
        network, center, scale = model
        normalized = (features - center) / np.where(scale > 1e-8, scale, 1.0)
        with torch.no_grad():
            scores = torch.sigmoid(network(torch.as_tensor(normalized, dtype=torch.float32))).cpu().numpy()
    else:
        raise ValueError(f"unknown synthetic classifier: {model_name}")
    result = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite {model_name} query scores")
    return result


def _model_features(runs: np.ndarray) -> np.ndarray:
    values = _runs_matrix(runs, name="runs")
    time = _resample_rows(values, 32)
    spectrum = np.log1p(np.abs(np.fft.rfft(values, axis=1)))
    frequency = _resample_rows(spectrum, 16)
    segment_means = np.asarray(
        [[float(np.mean(part)) for part in np.array_split(row, 8)] for row in values],
        dtype=float,
    )
    summaries = np.column_stack((
        np.mean(values, axis=1),
        np.std(values, axis=1),
        np.median(values, axis=1),
        np.median(np.abs(values - np.median(values, axis=1, keepdims=True)), axis=1),
        np.max(values, axis=1) - np.min(values, axis=1),
        np.mean(np.abs(np.diff(values, axis=1)), axis=1),
        np.mean(values * values, axis=1),
        np.max(np.abs(values), axis=1),
    ))
    features = np.concatenate((time, frequency, segment_means, summaries), axis=1)
    if features.shape != (len(values), FEATURE_WIDTH) or not np.isfinite(features).all():
        raise ValueError("synthetic model features are invalid")
    return features


def _perturb(
    row: np.ndarray,
    name: str,
    *,
    median: float,
    scale: float,
    config: SyntheticAnomalyConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    length = len(row)
    sign = float(rng.choice((-1.0, 1.0)))
    if name == "spike":
        result = row.copy()
        margin = max(1, length // 10)
        result[int(rng.integers(margin, max(margin + 1, length - margin)))] += sign * config.spike_mad_multiplier * scale
        return result
    if name == "drift":
        return row + sign * config.drift_mad_multiplier * scale * np.linspace(0.0, 1.0, length)
    if name == "level_shift":
        result = row.copy()
        result[int(round(length * 0.5)) :] += sign * config.level_shift_mad_multiplier * scale
        return result
    if name == "flat_line":
        result = row.copy()
        span = max(1, int(round(length * config.flat_line_fraction)))
        start = (length - span) // 2
        result[start : start + span] = median
        return result
    if name == "amplitude_scale":
        return median + (row - median) * config.amplitude_scale
    if name == "frequency_warp":
        positions = np.linspace(0.0, length - 1.0, length)
        source = np.mod(positions * config.frequency_warp_factor, length - 1.0)
        return np.interp(source, positions, row)
    if name == "phase_order":
        parts = np.array_split(row, int(config.phase_order_segments))
        shift = int(rng.integers(1, len(parts)))
        return np.concatenate(parts[shift:] + parts[:shift])
    raise ValueError(f"unknown synthetic perturbation: {name}")


def _empirical_p_values(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            (1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0)
            for value in query
        ],
        dtype=float,
    )


def _runs_matrix(values: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional run matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain finite normalized values")
    return matrix


def _resample_rows(values: np.ndarray, width: int) -> np.ndarray:
    source = np.arange(values.shape[1], dtype=float)
    positions = np.linspace(0.0, values.shape[1] - 1.0, int(width))
    return np.asarray([np.interp(positions, source, row) for row in values], dtype=float)


def _synthetic_sha256(by_type: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in SYNTHETIC_PERTURBATION_TYPES:
        values = np.ascontiguousarray(by_type[name], dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_config(config: SyntheticAnomalyConfig) -> FrozenConfig:
    if not isinstance(config, SyntheticAnomalyConfig):
        raise TypeError("config must be SyntheticAnomalyConfig")
    frozen = config.frozen
    if not isinstance(frozen, FrozenConfig):
        raise TypeError("SyntheticAnomalyConfig.frozen must be FrozenConfig")
    if int(frozen.seed) != frozen.seed or int(frozen.target_length) < 2:
        raise ValueError("frozen config is invalid")
    if tuple(frozen.alphas) != ALPHAS:
        raise ValueError("FrozenConfig.alphas must match the frozen alpha contract")
    if int(config.examples_per_type_per_run) != config.examples_per_type_per_run or config.examples_per_type_per_run < 1:
        raise ValueError("examples_per_type_per_run must be a positive integer")
    if not 0.0 < config.flat_line_fraction <= 1.0:
        raise ValueError("flat_line_fraction must be in (0, 1]")
    if config.amplitude_scale <= 0.0 or config.frequency_warp_factor <= 0.0:
        raise ValueError("synthetic scale factors must be positive")
    if int(config.phase_order_segments) != config.phase_order_segments or not 2 <= config.phase_order_segments <= frozen.target_length:
        raise ValueError("phase_order_segments must be in [2, target_length]")
    if int(config.mlp_epochs) != config.mlp_epochs or config.mlp_epochs < 1:
        raise ValueError("mlp_epochs must be a positive integer")
    if config.mlp_learning_rate <= 0.0:
        raise ValueError("mlp_learning_rate must be positive")
    return frozen


__all__ = [
    "MODEL_NAMES",
    "SYNTHETIC_PERTURBATION_TYPES",
    "SmallMLP",
    "SyntheticAnomalies",
    "SyntheticAnomalyConfig",
    "generate_synthetic_anomalies",
    "score_synthetic_models",
    "synthetic_shadow_predictions",
]
