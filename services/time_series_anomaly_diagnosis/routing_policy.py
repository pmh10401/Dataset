"""Anonymous TRAIN-feature routing for the live analysis policy."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .score_sources import METHOD_NAMES

FEATURE_KEYS = (
    "width",
    "n_train",
    "spectral_entropy",
    "skewness",
    "kurtosis",
    "diff_std_ratio",
    "zero_crossing_rate",
)
BASELINE_METHOD = METHOD_NAMES[0]


def extract_train_features(rows: Sequence[np.ndarray]) -> dict[str, float]:
    if len(rows) < 2:
        raise ValueError("at least two TRAIN rows are required")
    values = [np.asarray(row, dtype=np.float64) for row in rows]
    if any(row.ndim != 1 or row.size < 2 or not np.isfinite(row).all() for row in values):
        raise ValueError("TRAIN rows must be finite one-dimensional arrays of length at least two")

    entropies = []
    diff_ratios = []
    crossing_rates = []
    for row in values:
        centered = row - row.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2
        power_sum = float(power.sum())
        probabilities = power / (power_sum if power_sum > 0.0 else 1.0)
        log_probabilities = np.where(
            probabilities > 0.0, np.log(probabilities + 1e-300), 0.0
        )
        entropies.append(
            float(-(probabilities * log_probabilities).sum() / np.log(max(2, probabilities.size)))
        )
        diff_ratios.append(float(np.diff(row).std() / (row.std() + 1e-12)))
        crossing_rates.append(float(np.mean(np.diff(np.sign(centered)) != 0)))

    flat = np.concatenate(values)
    mean = float(flat.mean())
    standard_deviation = float(flat.std() + 1e-12)
    return {
        "width": float(np.median([row.size for row in values])),
        "n_train": float(len(values)),
        "spectral_entropy": float(np.median(entropies)),
        "skewness": float(np.mean((flat - mean) ** 3) / standard_deviation**3),
        "kurtosis": float(np.mean((flat - mean) ** 4) / standard_deviation**4 - 3.0),
        "diff_std_ratio": float(np.median(diff_ratios)),
        "zero_crossing_rate": float(np.median(crossing_rates)),
    }


@dataclass(frozen=True, slots=True)
class MethodMetric:
    method_name: str
    auc: float
    zf_true_positives: int

    def __post_init__(self) -> None:
        if self.method_name not in METHOD_NAMES:
            raise ValueError("method_name must be an approved score source")
        if not np.isfinite(self.auc) or not 0.0 <= self.auc <= 1.0:
            raise ValueError("auc must be finite and in [0, 1]")
        if type(self.zf_true_positives) is not int or self.zf_true_positives < 0:
            raise ValueError("zf_true_positives must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RouterTrainingRecord:
    group_id: int
    features: Mapping[str, float]
    metrics: tuple[MethodMetric, ...]

    def __post_init__(self) -> None:
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("group_id must be a non-negative integer")
        if tuple(self.features) != FEATURE_KEYS:
            raise ValueError("features must contain exactly the seven frozen inputs in order")
        if tuple(metric.method_name for metric in self.metrics) != METHOD_NAMES:
            raise ValueError("metrics must use the three approved methods in order")
        object.__setattr__(
            self, "features", MappingProxyType({key: float(self.features[key]) for key in FEATURE_KEYS})
        )


@dataclass(frozen=True, slots=True)
class CandidateGateEvidence:
    method_name: str
    passed: bool
    auc_non_degradation_rate: float
    zf_non_degradation_rate: float
    auc_gain_quantile: float
    mean_zf_gain: float


@dataclass(frozen=True, slots=True)
class RouterDecision:
    selected_method: str
    support: int
    candidate_evidence: tuple[CandidateGateEvidence, ...]


def _feature_vector(features: Mapping[str, float]) -> np.ndarray:
    if len(features) != len(FEATURE_KEYS) or set(features) != set(FEATURE_KEYS):
        raise ValueError("features must contain exactly the seven frozen inputs in order")
    values = np.asarray([float(features[key]) for key in FEATURE_KEYS], dtype=np.float64)
    for index in (0, 1):
        if values[index] > 0.0:
            values[index] = np.log1p(values[index])
    return values


@dataclass(frozen=True, slots=True)
class TrainFeatureRouter:
    input_keys: tuple[str, ...]
    _records: tuple[RouterTrainingRecord, ...]
    _medians: np.ndarray
    _means: np.ndarray
    _scales: np.ndarray
    _matrix: np.ndarray

    def route(self, features: Mapping[str, float]) -> RouterDecision:
        query = _feature_vector(features)
        query = np.where(np.isfinite(query), query, self._medians)
        query = (query - self._means) / self._scales
        distances = np.linalg.norm(self._matrix - query, axis=1)
        ordered = np.argsort(distances, kind="stable")
        selected = []
        groups = set()
        for index in ordered:
            group_id = self._records[int(index)].group_id
            if group_id not in groups:
                groups.add(group_id)
                selected.append(int(index))
                if len(selected) == 31:
                    break
        if len(selected) < 20:
            raise ValueError("router requires at least 20 anonymous groups")

        evidence = tuple(self._candidate_evidence(method, selected) for method in METHOD_NAMES[1:])
        passed = [item for item in evidence if item.passed]
        if not passed:
            method = BASELINE_METHOD
        else:
            fixed_order = {name: -index for index, name in enumerate(METHOD_NAMES[1:])}
            method = max(
                passed,
                key=lambda item: (
                    item.mean_zf_gain,
                    item.auc_gain_quantile,
                    fixed_order[item.method_name],
                ),
            ).method_name
        return RouterDecision(method, len(selected), evidence)

    def _candidate_evidence(
        self, method_name: str, selected: list[int]
    ) -> CandidateGateEvidence:
        auc_gains = []
        zf_gains = []
        for index in selected:
            metrics = {metric.method_name: metric for metric in self._records[index].metrics}
            auc_gains.append(metrics[method_name].auc - metrics[BASELINE_METHOD].auc)
            zf_gains.append(
                metrics[method_name].zf_true_positives
                - metrics[BASELINE_METHOD].zf_true_positives
            )
        auc = np.asarray(auc_gains, dtype=float)
        zf = np.asarray(zf_gains, dtype=float)
        auc_rate = float(np.mean(auc >= 0.0))
        zf_rate = float(np.mean(zf >= 0.0))
        quantile = float(np.quantile(auc, 0.20))
        mean_zf = float(np.mean(zf))
        passed = auc_rate >= 0.90 and zf_rate >= 0.90 and quantile >= 0.0 and mean_zf > 0.0
        return CandidateGateEvidence(method_name, passed, auc_rate, zf_rate, quantile, mean_zf)


def _protected(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.asarray(values, dtype=np.float64).tobytes(), dtype=np.float64)


def fit_train_feature_router(rows: Sequence[RouterTrainingRecord]) -> TrainFeatureRouter:
    records = tuple(rows)
    if len({record.group_id for record in records}) < 20:
        raise ValueError("router requires at least 20 anonymous groups")
    matrix = np.stack([_feature_vector(record.features) for record in records])
    finite = np.isfinite(matrix)
    if np.any(~finite.any(axis=0)):
        raise ValueError("each feature column requires a finite training value")
    medians = np.asarray(
        [np.median(matrix[finite[:, index], index]) for index in range(matrix.shape[1])]
    )
    matrix = np.where(finite, matrix, medians)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales == 0.0, 1.0, scales)
    return TrainFeatureRouter(
        FEATURE_KEYS,
        records,
        _protected(medians),
        _protected(means),
        _protected(scales),
        _protected((matrix - means) / scales).reshape(matrix.shape),
    )
