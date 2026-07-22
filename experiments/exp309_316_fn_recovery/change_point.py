from __future__ import annotations

import numpy as np

from .config import CHANGE_WINDOW_FRACTIONS
from .contracts import ScoreArtifact
from .foundation import crossfit_plan


def _windows(length: int) -> tuple[int, ...]:
    return tuple(sorted({int(np.clip(round(length * fraction), 4, max(4, length // 4))) for fraction in CHANGE_WINDOW_FRACTIONS}))


def _raw_features(series: np.ndarray, windows: tuple[int, ...]) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    values = np.asarray(series, dtype=np.float64).reshape(-1)
    features = []
    evidence = []
    for window in windows:
        best = (-np.inf, 0, 0.0, 0.0, 0.0)
        for point in range(window, len(values) - window + 1, max(1, window // 2)):
            left, right = values[point - window:point], values[point:point + window]
            location = abs(float(np.median(right) - np.median(left)))
            left_var, right_var = float(np.var(left) + 1e-6), float(np.var(right) + 1e-6)
            variance = abs(float(np.log(right_var / left_var)))
            quantile = float(np.mean(np.abs(np.quantile(left, [0.1, 0.5, 0.9]) - np.quantile(right, [0.1, 0.5, 0.9]))))
            combined = location + 0.25 * variance + quantile
            if combined > best[0]:
                best = (combined, point, location, variance, quantile)
        features.extend([best[2], best[3], best[4]])
        evidence.append({"window": int(window), "point": int(best[1]), "raw_score": float(best[0])})
    return np.asarray(features, dtype=np.float64), evidence


def score_change_point(train_normal: np.ndarray, test_values: np.ndarray, seed: int) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32)
    test = np.asarray(test_values, dtype=np.float32)
    if len(train) < 2:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"candidate_grain": "test_instance_virtual_run"}, False, "n_train<2")
    windows = _windows(train.shape[1])
    train_features = np.vstack([_raw_features(row, windows)[0] for row in train])
    test_pairs = [_raw_features(row, windows) for row in test]
    test_features = np.vstack([pair[0] for pair in test_pairs])
    train_scores = np.zeros(len(train), dtype=np.float64)
    for fit, query in crossfit_plan(len(train), seed=seed):
        fold_center = np.median(train_features[fit], axis=0)
        fold_scale = np.median(np.abs(train_features[fit] - fold_center), axis=0) * 1.4826
        fold_scale = np.where(fold_scale > 1e-8, fold_scale, 1.0)
        train_scores[query] = ((train_features[query] - fold_center) / fold_scale).max(axis=1)
    center = np.median(train_features, axis=0)
    scale = np.median(np.abs(train_features - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, 1.0)
    test_z = (test_features - center) / scale
    return ScoreArtifact(
        train_scores, test_z.max(axis=1),
        {
            "candidate_grain": "test_instance_virtual_run",
            "window_lengths": list(windows),
            "winning_component": np.argmax(test_z, axis=1).astype(int).tolist(),
            "window_evidence": [pair[1] for pair in test_pairs],
        },
    )
