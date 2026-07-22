from __future__ import annotations

import numpy as np
import pywt

from .config import WAVELET_SCALES
from .contracts import ScoreArtifact
from .foundation import crossfit_plan


def _features(series: np.ndarray, scales: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    coefficients, _ = pywt.cwt(values, scales, "cmor1.5-1.0")
    energy = np.log1p(np.mean(np.abs(coefficients) ** 2, axis=1))
    phase = np.unwrap(np.angle(coefficients), axis=1)
    increments = np.diff(phase, axis=1)
    phase_variance = 1.0 - np.abs(np.mean(np.exp(1j * increments), axis=1))
    phase_entropy = []
    for row in increments:
        hist, _ = np.histogram(((row + np.pi) % (2 * np.pi)) - np.pi, bins=12, range=(-np.pi, np.pi), density=False)
        probabilities = hist / max(1, hist.sum())
        nonzero = probabilities[probabilities > 0]
        phase_entropy.append(float(-np.sum(nonzero * np.log(nonzero))))
    coherence = []
    for index in range(len(scales) - 1):
        a, b = coefficients[index], coefficients[index + 1]
        coherence.append(float(abs(np.vdot(a, b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)))
    return np.concatenate([energy, phase_variance, phase_entropy, coherence]).astype(np.float64)


def score_wavelet_phase(train_normal: np.ndarray, test_values: np.ndarray, seed: int) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32)
    test = np.asarray(test_values, dtype=np.float32)
    if len(train) < 2:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"candidate_grain": "test_instance_virtual_run"}, False, "n_train<2")
    scales = tuple(scale for scale in WAVELET_SCALES if scale < train.shape[1])
    if len(scales) < 2:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"scales": list(scales)}, False, "target_length_too_short")
    train_features = np.vstack([_features(row, scales) for row in train])
    test_features = np.vstack([_features(row, scales) for row in test])
    train_scores = np.zeros(len(train), dtype=np.float64)
    for fit, query in crossfit_plan(len(train), seed=seed):
        fold_center = np.median(train_features[fit], axis=0)
        fold_spread = np.median(np.abs(train_features[fit] - fold_center), axis=0) * 1.4826
        fold_spread = np.where(fold_spread > 1e-8, fold_spread, 1.0)
        train_scores[query] = np.abs((train_features[query] - fold_center) / fold_spread).max(axis=1)
    center = np.median(train_features, axis=0)
    spread = np.median(np.abs(train_features - center), axis=0) * 1.4826
    spread = np.where(spread > 1e-8, spread, 1.0)
    test_z = np.abs((test_features - center) / spread)
    return ScoreArtifact(
        train_scores, test_z.max(axis=1),
        {
            "candidate_grain": "test_instance_virtual_run",
            "scales": list(scales),
            "winning_feature": np.argmax(test_z, axis=1).astype(int).tolist(),
        },
    )
