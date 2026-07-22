from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


def _mad(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float).reshape(-1)
    return float(1.4826 * np.median(np.abs(array - np.median(array))) + 1e-6)


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


class _StateSpaceModel:
    def __init__(self, reference: np.ndarray):
        first = np.diff(reference, axis=1)
        second = np.diff(first, axis=1)
        self.observation_variance = max(_mad(first) ** 2, 1e-6)
        self.level_variance = max(_mad(second) ** 2 * 0.25, 1e-7)
        self.trend_variance = max(_mad(second) ** 2 * 0.05, 1e-8)
        self.damping = 0.98

    def trace_features(self, row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=float)
        state = np.asarray([values[0], values[1] - values[0] if len(values) > 1 else 0.0])
        covariance = np.eye(2) * self.observation_variance
        transition = np.asarray([[1.0, self.damping], [0.0, self.damping]])
        observation = np.asarray([1.0, 0.0])
        process = np.diag([self.level_variance, self.trend_variance])
        innovations = []
        for value in values[1:]:
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            residual = float(value - observation @ state)
            variance = float(observation @ covariance @ observation + self.observation_variance)
            innovations.append(abs(residual) / np.sqrt(max(variance, 1e-12)))
            gain = covariance @ observation / max(variance, 1e-12)
            state = state + gain * residual
            covariance = covariance - np.outer(gain, observation) @ covariance
        stable = np.asarray(innovations[4:] if len(innovations) > 4 else innovations, dtype=float)
        if not len(stable):
            stable = np.zeros(1)
        return np.asarray(
            [
                np.median(stable),
                np.quantile(stable, 0.95),
                np.max(stable),
                _longest_run(stable > 3.0) / max(1, len(stable)),
            ],
            dtype=float,
        )

    def score(self, rows: np.ndarray) -> np.ndarray:
        features = np.vstack([self.trace_features(row) for row in rows])
        return features @ np.asarray([0.2, 0.3, 0.4, 0.1])


class RobustStateSpaceInnovation:
    name = "robust_state_space_innovation"
    minimum_train = 10

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train_oof = np.empty(len(bundle.train_normal), dtype=float)
        for fit, query in crossfit_plan(len(bundle.train_normal), seed=seed):
            train_oof[query] = _StateSpaceModel(bundle.train_normal[fit]).score(bundle.train_normal[query])
        model = _StateSpaceModel(bundle.train_normal)
        test_scores = model.score(bundle.test_values)
        return ScoreResult(
            self.name,
            train_oof,
            test_scores,
            False,
            None,
            {
                "seed": seed,
                "model": "numpy_local_level_damped_trend",
                "damping": model.damping,
                "cross_fitted": True,
                "self_reference_violations": 0,
                "contains_test_labels": False,
            },
        )

