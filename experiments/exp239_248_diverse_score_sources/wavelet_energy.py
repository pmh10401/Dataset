from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.source_base import feature_distance_result


def _entropy(probabilities: np.ndarray) -> float:
    values = probabilities[probabilities > 0]
    return float(-np.sum(values * np.log(values + 1e-12)))


def wavelet_features(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    wavelet = pywt.Wavelet("db4")
    level = max(1, min(5, pywt.dwt_max_level(values.shape[1], wavelet.dec_len)))
    features = []
    for row in values:
        coefficients = pywt.wavedec(row, wavelet, level=level)
        total = sum(float(np.sum(part * part)) for part in coefficients) + 1e-12
        vector = []
        for part in coefficients:
            energy = np.asarray(part, dtype=float) ** 2
            probability = energy / (energy.sum() + 1e-12)
            vector.extend(
                [
                    float(energy.sum() / total),
                    float(energy.max(initial=0.0) / total),
                    _entropy(probability),
                    float(np.max(probability, initial=0.0)),
                ]
            )
        features.append(vector)
    return np.asarray(features, dtype=float)


class MultiscaleWaveletEnergyDeviation:
    name = "multiscale_wavelet_energy_deviation"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        return feature_distance_result(
            source=self.name,
            bundle=bundle,
            seed=seed,
            extractor=wavelet_features,
            extra_provenance={"wavelet": "db4", "maximum_level": 5},
        )

