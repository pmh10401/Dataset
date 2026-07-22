from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.covariance import LedoitWolf


ALPHA = 0.05
FOLD_SEED = 20260722
MIN_TRAIN = 20
CONDITION_LIMIT = 1e8
DISTANCES = ("diagonal_mahalanobis", "shrinkage_mahalanobis", "correlation", "spectral_wasserstein")


def fixed_features(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError("finite aligned 2D series with length >= 8 required")
    n, length = x.shape; t = np.linspace(-1.0, 1.0, length)
    centered_t = t - t.mean(); slope = (x @ centered_t) / np.sum(centered_t ** 2)
    diff = np.diff(x, axis=1)
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered[:, :-1] ** 2, axis=1)
    autocorr = np.divide(
        np.sum(centered[:, :-1] * centered[:, 1:], axis=1), denom,
        out=np.zeros(n), where=denom > 1e-12,
    )
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    power[:, 0] = 0.0; power_sum = power.sum(axis=1, keepdims=True)
    normalized_power = np.divide(power, power_sum, out=np.zeros_like(power), where=power_sum > 1e-12)
    freq = np.linspace(0.0, 1.0, power.shape[1])
    spectral_centroid = normalized_power @ freq
    spectral_entropy = -np.sum(normalized_power * np.log(normalized_power + 1e-12), axis=1)
    q = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T
    return np.column_stack([
        x.mean(axis=1), x.std(axis=1), q, x.min(axis=1), x.max(axis=1),
        np.ptp(x, axis=1), slope, np.mean(np.abs(diff), axis=1), diff.std(axis=1),
        autocorr, spectral_centroid, spectral_entropy,
    ])


def _trace_and_spectrum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    centered = x - x.mean(axis=1, keepdims=True); scale = x.std(axis=1, keepdims=True)
    traces = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 1e-12)
    power = np.abs(np.fft.rfft(traces, axis=1)) ** 2; power[:, 0] = 0.0
    total = power.sum(axis=1, keepdims=True)
    spectra = np.divide(power, total, out=np.zeros_like(power), where=total > 1e-12)
    return traces, spectra


def medoid_count(n_train: int) -> int:
    return 1 if n_train < 10 else 2 if n_train < 30 else 3


def farthest_first_medoids(features: np.ndarray, count: int) -> np.ndarray:
    x = np.asarray(features, dtype=float); count = min(int(count), len(x))
    center = np.median(x, axis=0)
    selected = [int(np.argmin(np.sum((x - center) ** 2, axis=1)))]
    while len(selected) < count:
        distance = np.min(np.stack([np.sum((x - x[index]) ** 2, axis=1) for index in selected]), axis=0)
        distance[selected] = -1.0; selected.append(int(np.argmax(distance)))
    return np.asarray(selected, dtype=int)


@dataclass(frozen=True)
class DistanceModel:
    median: np.ndarray
    scale: np.ndarray
    medoid_features: np.ndarray
    medoid_traces: np.ndarray
    medoid_spectra: np.ndarray
    precision: np.ndarray
    shrinkage_fallback: bool
    condition_number: float


def fit_distance_model(values: np.ndarray) -> DistanceModel:
    features = fixed_features(values); median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0) * 1.4826
    iqr = np.subtract(*np.quantile(features, [0.75, 0.25], axis=0)) / 1.349
    std = features.std(axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(iqr > 1e-9, iqr, np.where(std > 1e-9, std, 1.0)))
    z = (features - median) / scale; traces, spectra = _trace_and_spectrum(values)
    medoids = farthest_first_medoids(z, medoid_count(len(values)))
    fallback = False
    try:
        precision = LedoitWolf().fit(z).precision_
        condition = float(np.linalg.cond(precision))
        if not np.isfinite(precision).all() or not np.isfinite(condition) or condition > CONDITION_LIMIT:
            raise ValueError("unstable shrinkage precision")
    except (ValueError, np.linalg.LinAlgError):
        precision = np.eye(z.shape[1]); condition = 1.0; fallback = True
    return DistanceModel(median, scale, z[medoids], traces[medoids], spectra[medoids], precision, fallback, condition)


def score_distances(model: DistanceModel, values: np.ndarray) -> dict[str, np.ndarray]:
    features = fixed_features(values); z = (features - model.median) / model.scale
    traces, spectra = _trace_and_spectrum(values)
    delta = z[:, None, :] - model.medoid_features[None, :, :]
    diagonal = np.sqrt(np.min(np.sum(delta ** 2, axis=2), axis=1))
    shrink = np.sqrt(np.maximum(0.0, np.min(np.einsum("nkd,df,nkf->nk", delta, model.precision, delta), axis=1)))
    correlation_matrix = 1.0 - (traces @ model.medoid_traces.T) / traces.shape[1]
    query_constant = np.linalg.norm(traces, axis=1) <= 1e-12
    medoid_constant = np.linalg.norm(model.medoid_traces, axis=1) <= 1e-12
    correlation_matrix[np.outer(query_constant, medoid_constant)] = 0.0
    correlation_matrix[np.logical_xor(query_constant[:, None], medoid_constant[None, :])] = 1.0
    correlation = np.min(np.clip(correlation_matrix, 0.0, 2.0), axis=1)
    freq = np.linspace(0.0, 1.0, spectra.shape[1])
    spectral_values = []
    for row in spectra:
        distances = []
        for medoid in model.medoid_spectra:
            if row.sum() <= 1e-12 and medoid.sum() <= 1e-12:
                distances.append(0.0)
            elif row.sum() <= 1e-12 or medoid.sum() <= 1e-12:
                distances.append(1.0)
            else:
                distances.append(wasserstein_distance(freq, freq, u_weights=row, v_weights=medoid))
        spectral_values.append(min(distances))
    spectral = np.asarray(spectral_values)
    return dict(zip(DISTANCES, (diagonal, shrink, correlation, spectral), strict=True))


def cross_fitted_references(train: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    n = len(train)
    if n < MIN_TRAIN:
        raise ValueError("n_train_below_20")
    rng = np.random.default_rng(FOLD_SEED); order = rng.permutation(n); folds = np.array_split(order, 5)
    references = {name: np.empty(n, dtype=float) for name in DISTANCES}; fallbacks = 0
    for heldout in folds:
        fit_indices = np.setdiff1d(np.arange(n), heldout, assume_unique=True)
        model = fit_distance_model(train[fit_indices]); fallbacks += int(model.shrinkage_fallback)
        scores = score_distances(model, train[heldout])
        for name in DISTANCES: references[name][heldout] = scores[name]
    return references, fallbacks


def empirical_pvalue(reference: np.ndarray, value: float) -> float:
    ref = np.asarray(reference, dtype=float)
    return float((1 + np.count_nonzero(ref >= value)) / (len(ref) + 1))


__all__ = ["ALPHA", "DISTANCES", "cross_fitted_references", "empirical_pvalue", "fit_distance_model", "fixed_features", "score_distances"]
