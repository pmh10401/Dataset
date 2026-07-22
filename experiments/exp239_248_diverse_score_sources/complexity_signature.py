from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.source_base import feature_distance_result


_PERMUTATIONS = {value: index for index, value in enumerate(itertools.permutations(range(3)))}


def _entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    return float(-np.sum(values * np.log(values + 1e-12)))


def _permutation_entropy(row: np.ndarray) -> float:
    counts = np.zeros(len(_PERMUTATIONS), dtype=float)
    for start in range(max(0, len(row) - 2)):
        key = tuple(np.argsort(row[start : start + 3], kind="stable"))
        counts[_PERMUTATIONS[key]] += 1
    probabilities = counts / max(1.0, counts.sum())
    return _entropy(probabilities) / np.log(len(counts))


def complexity_features(rows: np.ndarray) -> np.ndarray:
    output = []
    for row in np.asarray(rows, dtype=float):
        delta = np.diff(row)
        second = np.diff(delta)
        spectrum = np.abs(np.fft.rfft(row)) ** 2
        spectrum = spectrum[1:] if len(spectrum) > 1 else spectrum
        probability = spectrum / (spectrum.sum() + 1e-12)
        activity = float(np.var(row))
        mobility = float(np.sqrt(np.var(delta) / (activity + 1e-12)))
        second_mobility = float(np.sqrt(np.var(second) / (np.var(delta) + 1e-12))) if len(second) else 0.0
        complexity = second_mobility / (mobility + 1e-12)
        output.append(
            [
                _permutation_entropy(row),
                _entropy(probability) / np.log(max(2, len(probability))),
                activity,
                mobility,
                complexity,
                float(np.median(np.abs(delta))) if len(delta) else 0.0,
                float(np.mean(row[:-1] * row[1:] < 0)) if len(row) > 1 else 0.0,
                float(np.mean(delta[:-1] * delta[1:] < 0)) if len(delta) > 1 else 0.0,
                float(np.mean(np.abs(delta) < 1e-6)) if len(delta) else 1.0,
            ]
        )
    return np.asarray(output, dtype=float)


class ComplexitySignatureDeviation:
    name = "complexity_signature_deviation"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        return feature_distance_result(
            source=self.name,
            bundle=bundle,
            seed=seed,
            extractor=complexity_features,
            extra_provenance={"feature_set": "complexity_signature_v1"},
        )

