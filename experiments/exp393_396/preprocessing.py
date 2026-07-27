from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.exp393_396.contracts import PreparedDataset


def sanitize_series(values: Sequence[float] | np.ndarray) -> np.ndarray:
    series = np.asarray(values, dtype=float).reshape(-1)
    if series.size == 0:
        return np.zeros(1, dtype=float)
    finite = np.isfinite(series)
    if not finite.any():
        return np.zeros_like(series)
    indices = np.arange(series.size, dtype=float)
    return np.interp(indices, indices[finite], series[finite])


def derive_target_length(train_series: Sequence[Sequence[float] | np.ndarray]) -> int:
    if not train_series:
        raise ValueError("at least one TRAIN series is required")
    return max(1, max(np.asarray(series).size for series in train_series))


def align_series(values: Sequence[float] | np.ndarray, target_length: int) -> np.ndarray:
    series = sanitize_series(values)
    if target_length < 1:
        raise ValueError("target_length must be positive")
    if series.size == target_length:
        return series
    if series.size == 1:
        return np.full(target_length, series[0], dtype=float)
    source = np.linspace(0.0, 1.0, series.size)
    target = np.linspace(0.0, 1.0, target_length)
    return np.interp(target, source, series)


def prepare_dataset(
    train_series: Sequence[Sequence[float] | np.ndarray],
    test_series: Sequence[Sequence[float] | np.ndarray],
) -> PreparedDataset:
    target_length = derive_target_length(train_series)
    train = np.vstack(
        [align_series(series, target_length) for series in train_series]
    )
    if test_series:
        test = np.vstack(
            [align_series(series, target_length) for series in test_series]
        )
    else:
        test = np.empty((0, target_length), dtype=float)

    center = np.median(train, axis=0)
    absolute_deviation = np.abs(train - center)
    scale = np.median(absolute_deviation, axis=0) * 1.4826
    positive = scale[scale > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    scale = np.where(scale > 1e-12, scale, fallback)

    return PreparedDataset(
        train=(train - center) / scale,
        test=(test - center) / scale,
        target_length=target_length,
        train_center=center,
        train_scale=scale,
    )


def _spectral_entropy(window: np.ndarray) -> float:
    centered = window - np.mean(window)
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(power.sum())
    if total <= 1e-15:
        return 0.0
    probability = power / total
    probability = probability[probability > 0.0]
    entropy = -float(np.sum(probability * np.log(probability)))
    maximum = np.log(max(2, power.size))
    return entropy / maximum


def _lag_one_autocorrelation(window: np.ndarray) -> float:
    if window.size < 3:
        return 0.0
    left = window[:-1] - np.mean(window[:-1])
    right = window[1:] - np.mean(window[1:])
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _window_descriptor(window: np.ndarray) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, window.size)
    slope = float(np.polyfit(x, window, 1)[0]) if window.size > 1 else 0.0
    curvature = np.diff(window, n=2)
    curvature_energy = (
        float(np.mean(curvature**2)) if curvature.size else 0.0
    )
    median = float(np.median(window))
    mad = float(np.median(np.abs(window - median)))
    return np.asarray(
        [
            median,
            slope,
            curvature_energy,
            mad,
            _lag_one_autocorrelation(window),
            _spectral_entropy(window),
        ],
        dtype=float,
    )


def segment_descriptors(trace: np.ndarray, target_length: int) -> np.ndarray:
    values = align_series(trace, target_length)
    n_segments = min(32, max(8, target_length // 8))
    width = max(2, int(np.ceil(target_length / n_segments)))
    width = min(width, target_length)
    primary = np.linspace(
        0,
        max(0, target_length - width),
        num=n_segments,
        dtype=int,
    )
    offset = np.clip(primary + max(1, width // 2), 0, target_length - width)
    starts = np.unique(np.concatenate([primary, offset]))
    descriptors = [_window_descriptor(values[start : start + width]) for start in starts]
    return np.vstack(descriptors)
