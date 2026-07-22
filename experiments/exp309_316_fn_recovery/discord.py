from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .config import DISCORD_SCALE_FRACTIONS, MAX_PATCH_MEMORY
from .contracts import ScoreArtifact
from .foundation import crossfit_plan


def _window_lengths(length: int) -> tuple[int, ...]:
    upper = max(8, int(length) // 4)
    return tuple(sorted({int(np.clip(round(length * fraction), 8, upper)) for fraction in DISCORD_SCALE_FRACTIONS}))


def _patch_signatures(series: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(series, dtype=np.float32).reshape(-1)
    stride = max(1, int(window) // 2)
    starts = np.arange(0, max(1, len(values) - int(window) + 1), stride, dtype=int)
    if not len(starts):
        starts = np.asarray([0], dtype=int)
    target = np.linspace(0.0, 1.0, 8)
    signatures = []
    for start in starts:
        patch = values[int(start): int(start) + int(window)]
        if len(patch) < int(window):
            patch = np.pad(patch, (0, int(window) - len(patch)), mode="edge")
        x = np.linspace(0.0, 1.0, len(patch))
        shape = np.interp(target, x, patch)
        shape = (shape - shape.mean()) / max(float(shape.std()), 1e-6)
        signatures.append(np.concatenate([shape, [patch.mean(), patch.std()]]))
    return np.asarray(signatures, dtype=np.float32), starts


def _memory(rows: np.ndarray, window: int, seed: int) -> np.ndarray:
    signatures = np.vstack([_patch_signatures(row, window)[0] for row in rows])
    if len(signatures) > MAX_PATCH_MEMORY:
        chosen = np.random.default_rng(int(seed) + int(window)).choice(len(signatures), MAX_PATCH_MEMORY, replace=False)
        signatures = signatures[np.sort(chosen)]
    return signatures


def _score_rows(rows: np.ndarray, memory: np.ndarray, window: int) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    scores = []
    evidence = []
    for row in rows:
        signatures, starts = _patch_signatures(row, window)
        nearest = cdist(signatures, memory, metric="euclidean").min(axis=1)
        winning = int(np.argmax(nearest))
        scores.append(float(np.quantile(nearest, 0.95)))
        evidence.append({
            "start": int(starts[winning]),
            "end": int(min(len(row), starts[winning] + window)),
            "max_patch_distance": float(nearest[winning]),
        })
    return np.asarray(scores, dtype=np.float64), evidence


def score_discord(train_normal: np.ndarray, test_values: np.ndarray, seed: int) -> ScoreArtifact:
    train = np.asarray(train_normal, dtype=np.float32)
    test = np.asarray(test_values, dtype=np.float32)
    if len(train) < 2:
        return ScoreArtifact(np.asarray([]), np.zeros(len(test)), {"candidate_grain": "test_instance_virtual_run"}, False, "n_train<2")
    scales = _window_lengths(train.shape[1])
    train_by_scale = np.zeros((len(train), len(scales)), dtype=np.float64)
    test_by_scale = np.zeros((len(test), len(scales)), dtype=np.float64)
    evidence_by_scale: dict[int, list[dict[str, int | float]]] = {}
    for scale_index, window in enumerate(scales):
        for fold_index, (fit, query) in enumerate(crossfit_plan(len(train), seed=seed)):
            memory = _memory(train[fit], window, seed + fold_index)
            train_by_scale[query, scale_index] = _score_rows(train[query], memory, window)[0]
        full_memory = _memory(train, window, seed)
        test_scores, evidence = _score_rows(test, full_memory, window)
        test_by_scale[:, scale_index] = test_scores
        evidence_by_scale[int(window)] = evidence
    center = np.median(train_by_scale, axis=0)
    scale = np.median(np.abs(train_by_scale - center), axis=0) * 1.4826
    scale = np.where(scale > 1e-8, scale, 1.0)
    train_z = (train_by_scale - center) / scale
    test_z = (test_by_scale - center) / scale
    winning = np.argmax(test_z, axis=1)
    diagnostics = {
        "candidate_grain": "test_instance_virtual_run",
        "window_lengths": list(scales),
        "winning_window": [int(scales[index]) for index in winning],
        "winning_segment": [evidence_by_scale[int(scales[index])][row] for row, index in enumerate(winning)],
    }
    return ScoreArtifact(train_z.max(axis=1), test_z.max(axis=1), diagnostics)
