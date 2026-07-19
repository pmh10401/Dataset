from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf


EPSILON = 1e-8


def _rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 8:
        raise ValueError("score sources require a non-empty 2D matrix with length >= 8")
    if not np.isfinite(matrix).all():
        raise ValueError("score source input must be finite")
    return matrix


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    matrix = _rows(values)
    center = np.median(matrix, axis=1, keepdims=True)
    mad = np.median(np.abs(matrix - center), axis=1, keepdims=True)
    scale = 1.4826 * mad
    standard = np.std(matrix, axis=1, keepdims=True)
    scale = np.where(scale > EPSILON, scale, np.where(standard > EPSILON, standard, 1.0))
    return (matrix - center) / scale


def _resample(values: np.ndarray, length: int) -> np.ndarray:
    matrix = _rows(values)
    if matrix.shape[1] == length:
        return matrix
    old = np.linspace(0.0, 1.0, matrix.shape[1])
    new = np.linspace(0.0, 1.0, int(length))
    return np.vstack([np.interp(new, old, row) for row in matrix])


def _change_features(values: np.ndarray) -> np.ndarray:
    matrix = _robust_normalize(values)
    first = np.diff(matrix, axis=1)
    second = np.diff(first, axis=1)
    half = matrix.shape[1] // 2
    cumulative = np.cumsum(matrix - matrix.mean(axis=1, keepdims=True), axis=1)
    return np.column_stack(
        (
            np.quantile(np.abs(first), 0.95, axis=1),
            np.max(np.abs(first), axis=1),
            np.quantile(np.abs(second), 0.95, axis=1),
            np.max(np.abs(second), axis=1),
            np.median(matrix[:, half:], axis=1) - np.median(matrix[:, :half], axis=1),
            np.max(cumulative, axis=1) - np.min(cumulative, axis=1),
        )
    )


def _normal_space_features(values: np.ndarray) -> np.ndarray:
    matrix = _robust_normalize(values)
    first = np.diff(matrix, axis=1)
    quantiles = np.quantile(matrix, (0.05, 0.25, 0.5, 0.75, 0.95), axis=1).T
    lags = []
    for lag in (1, 2, 4, 8):
        actual = min(lag, matrix.shape[1] - 1)
        lags.append(np.mean(matrix[:, :-actual] * matrix[:, actual:], axis=1))
    spectrum = np.abs(np.fft.rfft(matrix, axis=1)) ** 2
    spectrum = spectrum[:, 1:]
    bands = []
    for indices in np.array_split(np.arange(spectrum.shape[1]), 6):
        bands.append(np.log1p(spectrum[:, indices].mean(axis=1)))
    return np.column_stack(
        (
            quantiles,
            matrix.mean(axis=1),
            matrix.std(axis=1),
            np.mean(np.abs(first), axis=1),
            np.max(np.abs(first), axis=1),
            *lags,
            *bands,
        )
    )


def _robust_feature_model(features: np.ndarray) -> dict[str, np.ndarray]:
    center = np.median(features, axis=0)
    mad = 1.4826 * np.median(np.abs(features - center), axis=0)
    std = np.std(features, axis=0)
    scale = np.where(mad > EPSILON, mad, np.where(std > EPSILON, std, 1.0))
    return {"center": center, "scale": scale}


def _robust_feature_score(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    z = np.abs((features - model["center"]) / model["scale"])
    return np.mean(np.sort(z, axis=1)[:, -min(3, z.shape[1]) :], axis=1)


def _ar_design(matrix: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [matrix[:, lag - offset - 1 : matrix.shape[1] - offset - 1] for offset in range(lag)],
        axis=2,
    )
    return x.reshape(-1, lag), matrix[:, lag:].reshape(-1)


def _shift_distance(rows: np.ndarray, prototype: np.ndarray, max_shift: int = 4) -> np.ndarray:
    distances = []
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            left, right = rows[:, :shift], prototype[-shift:]
        elif shift > 0:
            left, right = rows[:, shift:], prototype[:-shift]
        else:
            left, right = rows, prototype
        distances.append(np.mean((left - right) ** 2, axis=1))
    return np.min(np.vstack(distances), axis=0)


def _subsequences(values: np.ndarray, *, row_limit: int | None = None) -> np.ndarray:
    matrix = _resample(_robust_normalize(values), 64)
    if row_limit is not None and len(matrix) > row_limit:
        chosen = np.linspace(0, len(matrix) - 1, row_limit, dtype=int)
        matrix = matrix[chosen]
    starts = np.linspace(0, 48, 7, dtype=int)
    windows = np.vstack([matrix[:, start : start + 16] for start in starts])
    center = windows.mean(axis=1, keepdims=True)
    scale = windows.std(axis=1, keepdims=True)
    return (windows - center) / np.where(scale > EPSILON, scale, 1.0)


@dataclass(frozen=True)
class SourceModel:
    name: str
    payload: dict[str, Any]


def fit_score_source(name: str, train: np.ndarray) -> SourceModel:
    matrix = _rows(train)
    if name == "pooled_ar_residual":
        normalized = _robust_normalize(matrix)
        lag = min(8, max(1, normalized.shape[1] // 8))
        xtx = np.zeros((lag, lag), dtype=np.float64)
        xty = np.zeros(lag, dtype=np.float64)
        for start in range(0, len(normalized), 256):
            design, target = _ar_design(normalized[start : start + 256], lag)
            xtx += design.T @ design
            xty += design.T @ target
        ridge = 1e-3 * np.eye(lag)
        coefficients = np.linalg.solve(xtx + ridge, xty)
        return SourceModel(name, {"lag": lag, "coefficients": coefficients})
    if name == "robust_change_statistics":
        return SourceModel(name, _robust_feature_model(_change_features(matrix)))
    if name == "shrinkage_normal_space":
        features = _normal_space_features(matrix)
        standardizer = _robust_feature_model(features)
        z = (features - standardizer["center"]) / standardizer["scale"]
        covariance = LedoitWolf().fit(z)
        return SourceModel(
            name,
            {
                **standardizer,
                "location": covariance.location_,
                "precision": covariance.precision_,
            },
        )
    if name == "elastic_normal_prototype":
        normalized = _resample(_robust_normalize(matrix), 64)
        if len(normalized) > 32:
            normalized = normalized[np.linspace(0, len(normalized) - 1, 32, dtype=int)]
        distances = np.mean((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2)
        prototype = normalized[int(np.argmin(distances.sum(axis=1)))]
        return SourceModel(name, {"prototype": prototype})
    if name == "subsequence_discord":
        dictionary = _subsequences(matrix, row_limit=16)
        return SourceModel(name, {"dictionary": dictionary})
    raise ValueError(f"unknown score source: {name}")


def score_rows(model: SourceModel, values: np.ndarray) -> np.ndarray:
    matrix = _rows(values)
    if model.name == "pooled_ar_residual":
        normalized = _robust_normalize(matrix)
        output = np.empty(len(normalized), dtype=float)
        for start in range(0, len(normalized), 256):
            chunk = normalized[start : start + 256]
            design, target = _ar_design(chunk, int(model.payload["lag"]))
            residual = (target - design @ model.payload["coefficients"]).reshape(len(chunk), -1)
            output[start : start + len(chunk)] = np.sqrt(np.mean(residual**2, axis=1))
        return output
    if model.name == "robust_change_statistics":
        return _robust_feature_score(model.payload, _change_features(matrix))
    if model.name == "shrinkage_normal_space":
        features = _normal_space_features(matrix)
        z = (features - model.payload["center"]) / model.payload["scale"]
        delta = z - model.payload["location"]
        return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, model.payload["precision"], delta)))
    if model.name == "elastic_normal_prototype":
        normalized = _resample(_robust_normalize(matrix), 64)
        return np.sqrt(_shift_distance(normalized, model.payload["prototype"]))
    if model.name == "subsequence_discord":
        dictionary = model.payload["dictionary"]
        output = np.empty(len(matrix), dtype=float)
        for start in range(0, len(matrix), 128):
            chunk = matrix[start : start + 128]
            windows = _subsequences(chunk)
            per_window = np.sqrt(
                np.min(
                    np.mean((windows[:, None, :] - dictionary[None, :, :]) ** 2, axis=2),
                    axis=1,
                )
            )
            output[start : start + len(chunk)] = np.max(
                per_window.reshape(7, len(chunk)).T, axis=1
            )
        return output
    raise ValueError(f"unknown score source: {model.name}")
