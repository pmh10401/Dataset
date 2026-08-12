"""Copied canonical normal-state features and deterministic cross-fitting.

Provenance: experiments/exp177_185/core.py (canonical_features,
_multiscale_stat_features, _fold_plan, _score_fit_query, crossfit_scores),
SHA256 49f15e39c39a801842abca43cf67f73b79ded0a273029071e079f29100d2cc43.
"""
from __future__ import annotations

import numpy as np


CANONICAL_SEED_PAPER = 20260718
PROVENANCE_PAPER = {"experiments/exp177_185/core.py": ("canonical_features", "_multiscale_stat_features", "_fold_plan", "_score_fit_query", "crossfit_scores")}


def _sanitize_matrix_paper(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64)
    if result.ndim != 2: raise ValueError("feature input must be a matrix")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _multiscale_stat_features_paper(matrix: np.ndarray) -> np.ndarray:
    rows = []
    for row in _sanitize_matrix_paper(matrix):
        centered = row - np.mean(row); scale = float(np.std(centered)) or 1.0; normalized = centered / scale; first, second = np.diff(normalized), np.diff(np.diff(normalized))
        values = [float(np.mean(row)), float(np.std(row)), *[float(np.quantile(row, q)) for q in (.05, .25, .5, .75, .95)], float(np.mean(np.abs(first))) if len(first) else 0., float(np.std(first)) if len(first) else 0., float(np.std(second)) if len(second) else 0., float(np.mean(np.signbit(normalized[:-1]) != np.signbit(normalized[1:]))) if len(normalized) > 1 else 0.]
        values.extend(float(np.mean(normalized[:-lag] * normalized[lag:])) if len(normalized) > lag else 0. for lag in (1, 2, 4, 8, 16, 32))
        power = np.abs(np.fft.rfft(normalized)) ** 2; probabilities = power / (float(np.sum(power)) or 1.); frequencies = np.linspace(0., 1., len(power)); values.extend((float(np.sum(probabilities * frequencies)), float(-np.sum(probabilities * np.log(probabilities + 1e-12)))))
        values.extend(float(np.sum(probabilities[(frequencies >= low) & (frequencies < high)])) for low, high in ((0., .1), (.1, .25), (.25, .5), (.5, 1.01)))
        rows.append(values)
    return np.asarray(rows, dtype=np.float64)


def canonical_features_paper(matrix: np.ndarray) -> np.ndarray:
    try: from pycatch22 import catch22_all
    except ImportError as exc: raise RuntimeError("dependency_unavailable:pycatch22") from exc
    values = _sanitize_matrix_paper(matrix); catch = [np.asarray(catch22_all(row.tolist(), catch24=True)["values"], dtype=float) for row in values]
    return np.nan_to_num(np.hstack((np.vstack(catch), _multiscale_stat_features_paper(values))), nan=0., posinf=0., neginf=0.)


def _fold_plan_paper(count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if count < 5: raise ValueError("crossfit requires at least five TRAIN-normal instances")
    order = np.random.default_rng(seed).permutation(count); all_indices = np.arange(count)
    return [(np.setdiff1d(all_indices, query, assume_unique=True), np.sort(query)) for query in np.array_split(order, min(5, count)) if len(query)]


def _score_fit_query_paper(fit: np.ndarray, query: np.ndarray, backend: str) -> np.ndarray:
    center = np.median(fit, axis=0); scale = 1.4826 * np.median(np.abs(fit - center), axis=0); std = np.std(fit, axis=0); scale = np.where(scale > 1e-9, scale, np.where(std > 1e-9, std, 1.0)); fit_z, query_z = (fit - center) / scale, (query - center) / scale
    if backend != "knn_local_gap": raise ValueError(f"unknown score backend: {backend}")
    k = max(1, min(5, len(fit_z))); distances = np.sqrt(np.mean((query_z[:, None] - fit_z[None]) ** 2, axis=2)); query_distance = np.mean(np.partition(distances, k - 1, axis=1)[:, :k], axis=1)
    if len(fit_z) <= 1: return query_distance
    fit_distances = np.sqrt(np.mean((fit_z[:, None] - fit_z[None]) ** 2, axis=2)); np.fill_diagonal(fit_distances, np.inf); fit_k = max(1, min(k, len(fit_z) - 1)); local = np.mean(np.partition(fit_distances, fit_k - 1, axis=1)[:, :fit_k], axis=1)
    return query_distance / (float(np.median(local)) or 1.0)


def crossfit_scores_paper(train_features: np.ndarray, test_features: np.ndarray, *, backend: str, seed: int = CANONICAL_SEED_PAPER) -> tuple[np.ndarray, np.ndarray]:
    train, test = np.asarray(train_features, dtype=float), np.asarray(test_features, dtype=float); oof = np.full(len(train), np.nan)
    for fit, query in _fold_plan_paper(len(train), seed): oof[query] = _score_fit_query_paper(train[fit], train[query], backend)
    test_scores = _score_fit_query_paper(train, test, backend)
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_scores)): raise ValueError("non-finite crossfit score")
    return oof, test_scores
