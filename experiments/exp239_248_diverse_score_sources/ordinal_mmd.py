from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


COMBINATIONS = ((3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 4))


def ordinal_features(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    vectors = []
    mappings = {
        dimension: {value: index for index, value in enumerate(itertools.permutations(range(dimension)))}
        for dimension in (3, 4)
    }
    for row in values:
        vector = []
        for dimension, delay in COMBINATIONS:
            counts = np.zeros(len(mappings[dimension]), dtype=float)
            width = (dimension - 1) * delay + 1
            for start in range(max(0, len(row) - width + 1)):
                window = row[start : start + width : delay]
                counts[mappings[dimension][tuple(np.argsort(window, kind="stable"))]] += 1
            vector.extend(counts / max(1.0, counts.sum()))
        vectors.append(vector)
    return np.asarray(vectors, dtype=float)


def _bandwidth(reference: np.ndarray) -> float:
    delta = reference[:, None, :] - reference[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def _mmd_scores(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    bandwidth = max(_bandwidth(reference), 1e-6)
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    rr = reference[:, None, :] - reference[None, :, :]
    reference_term = float(np.exp(-gamma * np.sum(rr * rr, axis=2)).mean())
    scores = []
    for value in query:
        distance = reference - value
        cross = float(np.exp(-gamma * np.sum(distance * distance, axis=1)).mean())
        scores.append(np.sqrt(max(0.0, 1.0 - 2.0 * cross + reference_term)))
    return np.asarray(scores, dtype=float)


class OrdinalPatternDistributionShift:
    name = "ordinal_pattern_distribution_shift"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train_features = ordinal_features(bundle.train_normal)
        test_features = ordinal_features(bundle.test_values)
        train_oof = np.empty(len(train_features), dtype=float)
        for fit, query in crossfit_plan(len(train_features), seed=seed):
            train_oof[query] = _mmd_scores(train_features[fit], train_features[query])
        test_scores = _mmd_scores(train_features, test_features)
        return ScoreResult(
            self.name,
            train_oof,
            test_scores,
            False,
            None,
            {
                "seed": seed,
                "combinations": COMBINATIONS,
                "cross_fitted": True,
                "self_reference_violations": 0,
                "contains_test_labels": False,
            },
        )

