from __future__ import annotations

import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.exp282_284_prototype_distance_verifier.distance_engine import fixed_features


SEED = 20260722
MAX_BASES = 512
MIN_TRAIN = 100
ALPHAS = (0.01, 0.02, 0.05)


def synthetic_traces(values: np.ndarray, *, seed: int = SEED) -> tuple[np.ndarray, tuple[str, ...]]:
    x = np.asarray(values, dtype=float); rng = np.random.default_rng(seed); generated = []; labels = []
    for row in x:
        median = np.median(row); scale = np.median(np.abs(row - median)) * 1.4826
        if scale <= 1e-9: scale = row.std()
        if scale <= 1e-9: scale = 1.0
        sign = float(rng.choice((-1.0, 1.0)))
        spike = row.copy(); position = int(rng.integers(max(1, len(row) // 10), max(2, len(row) - len(row) // 10)))
        spike[position] += sign * 5.0 * scale; generated.append(spike); labels.append("spike")
        shift = row.copy(); shift[len(row) // 2:] += sign * 3.0 * scale; generated.append(shift); labels.append("level_shift")
        drift = row + sign * np.linspace(0.0, 4.0 * scale, len(row)); generated.append(drift); labels.append("drift")
        frequency = row + sign * 2.0 * scale * np.sin(np.linspace(0.0, 14.0 * np.pi, len(row), endpoint=False))
        generated.append(frequency); labels.append("frequency")
    return np.asarray(generated, dtype=float), tuple(labels)


def _selected(values: np.ndarray, identity: str) -> np.ndarray:
    if len(values) <= MAX_BASES: return np.asarray(values, dtype=float)
    derived = int.from_bytes(hashlib.sha256(f"{identity}:{SEED}".encode()).digest()[:8], "big")
    indices = np.random.default_rng(derived).choice(len(values), MAX_BASES, replace=False)
    return np.asarray(values, dtype=float)[np.sort(indices)]


def fit_classifier(train: np.ndarray, *, identity: str):
    normal = _selected(train, identity); synthetic, _ = synthetic_traces(normal, seed=SEED)
    x = np.vstack([fixed_features(normal), fixed_features(synthetic)])
    y = np.concatenate([np.zeros(len(normal), dtype=int), np.ones(len(synthetic), dtype=int)])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=SEED))
    model.fit(x, y); return model, len(normal), len(synthetic)


def anomaly_scores(model, values: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(fixed_features(values))[:, 1], dtype=float)


def cross_fitted_normal_scores(train: np.ndarray, *, identity: str) -> np.ndarray:
    n = len(train)
    if n < MIN_TRAIN: raise ValueError("n_train_below_100")
    order = np.random.default_rng(SEED).permutation(n); folds = np.array_split(order, 5); scores = np.empty(n)
    for fold_index, heldout in enumerate(folds):
        fit_indices = np.setdiff1d(np.arange(n), heldout, assume_unique=True)
        model, _, _ = fit_classifier(train[fit_indices], identity=f"{identity}:fold{fold_index}")
        scores[heldout] = anomaly_scores(model, train[heldout])
    return scores


def empirical_pvalue(reference: np.ndarray, value: float) -> float:
    return float((1 + np.count_nonzero(np.asarray(reference) >= value)) / (len(reference) + 1))


__all__ = ["ALPHAS", "MIN_TRAIN", "anomaly_scores", "cross_fitted_normal_scores", "empirical_pvalue", "fit_classifier", "synthetic_traces"]
