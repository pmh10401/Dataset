from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.robust_distance import RobustDistanceModel
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result, resample_rows


def _delay(reference: np.ndarray) -> int:
    mean_trace = np.mean(reference, axis=0)
    centered = mean_trace - mean_trace.mean()
    denominator = float(np.dot(centered, centered)) + 1e-12
    correlations = [float(np.dot(centered[:-lag], centered[lag:]) / denominator) for lag in range(1, min(17, len(centered)))]
    for lag, value in enumerate(correlations, 1):
        if value <= np.exp(-1):
            return lag
    return int(np.argmin(np.abs(correlations)) + 1) if correlations else 1


def _embedding(row: np.ndarray, dimension: int, delay: int) -> np.ndarray:
    count = len(row) - (dimension - 1) * delay
    if count <= 1:
        return np.asarray(row[:1], dtype=float).reshape(1, 1)
    return np.column_stack([row[offset * delay : offset * delay + count] for offset in range(dimension)])


def _threshold(reference: np.ndarray, delay: int) -> float:
    distances = []
    for row in reference[: min(20, len(reference))]:
        embedded = _embedding(row, 2, delay)
        if len(embedded) > 64:
            embedded = embedded[np.linspace(0, len(embedded) - 1, 64).astype(int)]
        delta = embedded[:, None, :] - embedded[None, :, :]
        matrix = np.sqrt(np.sum(delta * delta, axis=2))
        distances.extend(matrix[np.triu_indices(len(matrix), 1)])
    positive = np.asarray(distances, dtype=float)
    positive = positive[positive > 0]
    return float(np.quantile(positive, 0.1)) if len(positive) else 1e-6


def _run_lengths(values: np.ndarray) -> list[int]:
    lengths = []
    current = 0
    for value in values.astype(bool):
        if value:
            current += 1
        elif current:
            lengths.append(current); current = 0
    if current:
        lengths.append(current)
    return lengths


def _line_entropy(lengths: list[int]) -> float:
    selected = np.asarray([value for value in lengths if value >= 2], dtype=int)
    if not len(selected):
        return 0.0
    counts = np.bincount(selected)[2:].astype(float)
    probability = counts[counts > 0] / counts.sum()
    return float(-np.sum(probability * np.log(probability + 1e-12)))


def recurrence_features(rows: np.ndarray, delay: int, threshold: float) -> np.ndarray:
    output = []
    for row in rows:
        vector = []
        for dimension in (2, 3):
            embedded = _embedding(row, dimension, delay)
            delta = embedded[:, None, :] - embedded[None, :, :]
            recurrence = np.sqrt(np.sum(delta * delta, axis=2)) <= threshold
            np.fill_diagonal(recurrence, False)
            points = max(1, int(recurrence.sum()))
            diagonals = [_run_lengths(np.diagonal(recurrence, offset)) for offset in range(-len(recurrence) + 1, len(recurrence))]
            diagonal_lengths = [value for group in diagonals for value in group]
            vertical_lengths = [value for column in recurrence.T for value in _run_lengths(column)]
            diagonal_points = sum(value for value in diagonal_lengths if value >= 2)
            vertical_selected = [value for value in vertical_lengths if value >= 2]
            vector.extend(
                [
                    float(recurrence.mean()),
                    diagonal_points / points,
                    sum(vertical_selected) / points,
                    float(np.mean(vertical_selected)) if vertical_selected else 0.0,
                    _line_entropy(diagonal_lengths),
                    _line_entropy(vertical_lengths),
                ]
            )
        output.append(vector)
    return np.asarray(output, dtype=float)


class RecurrenceDynamicsDeviation:
    name = "recurrence_dynamics_deviation"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train = resample_rows(bundle.train_normal, 256)
        test = resample_rows(bundle.test_values, 256)
        train_oof = np.empty(len(train), dtype=float)
        fold_parameters = []
        for fit, query in crossfit_plan(len(train), seed=seed):
            delay = _delay(train[fit]); threshold = _threshold(train[fit], delay)
            model = RobustDistanceModel(recurrence_features(train[fit], delay, threshold))
            train_oof[query] = model.score(recurrence_features(train[query], delay, threshold))
            fold_parameters.append((delay, threshold))
        delay = _delay(train); threshold = _threshold(train, delay)
        model = RobustDistanceModel(recurrence_features(train, delay, threshold))
        test_scores = model.score(recurrence_features(test, delay, threshold))
        return ScoreResult(
            self.name,
            train_oof,
            test_scores,
            False,
            None,
            {
                "seed": seed,
                "maximum_length": 256,
                "delay": delay,
                "threshold": threshold,
                "fold_parameters": fold_parameters,
                "cross_fitted": True,
                "self_reference_violations": 0,
                "contains_test_labels": False,
            },
        )
