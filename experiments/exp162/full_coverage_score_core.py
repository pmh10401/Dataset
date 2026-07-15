from __future__ import annotations

import hashlib
import json
import numpy as np
from sklearn.preprocessing import RobustScaler
from numpy.lib.stride_tricks import sliding_window_view

def z_normalize(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std

def resample_rows(X: np.ndarray, target_len: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("resample_rows expects a 2D array")
    target_len = max(8, int(target_len))
    if X.shape[1] == target_len:
        return X.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, X.shape[1], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.vstack([np.interp(new_x, old_x, row).astype(np.float32) for row in X])

def spectral_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    d1 = np.diff(X, axis=1, prepend=X[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    fft = np.abs(np.fft.rfft(X, axis=1))
    if fft.shape[1] > 1:
        fft[:, 0] = 0.0
    total = fft.sum(axis=1, keepdims=True) + 1e-9
    norm_fft = fft / total
    n_bins = fft.shape[1]
    edges = np.linspace(0, n_bins, 9, dtype=int)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        bands.append(norm_fft[:, lo:hi].sum(axis=1))
    bands = np.vstack(bands).T
    freqs = np.arange(n_bins, dtype=np.float64)
    centroid = (norm_fft * freqs).sum(axis=1) / max(1, n_bins - 1)
    entropy = -(norm_fft * np.log(norm_fft + 1e-12)).sum(axis=1) / np.log(max(2, n_bins))
    peak = norm_fft.max(axis=1)

    def stats(A):
        return np.vstack([
            A.mean(axis=1),
            A.std(axis=1),
            np.median(A, axis=1),
            np.percentile(A, 90, axis=1) - np.percentile(A, 10, axis=1),
            np.max(np.abs(A), axis=1),
        ]).T

    return np.hstack([
        stats(X), stats(d1), stats(d2),
        bands,
        centroid[:, None], entropy[:, None], peak[:, None]
    ])

def stable_hash(name: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))

def choose_shapelets(X_train: np.ndarray, num_shapelets: int, lengths: list[int], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=np.float32)
    candidates = []
    valid_lengths = [length for length in lengths if 4 <= length <= X_train.shape[1]]
    if not valid_lengths:
        valid_lengths = [min(max(4, X_train.shape[1] // 4), X_train.shape[1])]
    for idx in range(int(num_shapelets)):
        row_idx = int(rng.integers(0, len(X_train)))
        length = int(valid_lengths[idx % len(valid_lengths)])
        start = int(rng.integers(0, max(1, X_train.shape[1] - length + 1)))
        shapelet = X_train[row_idx, start : start + length]
        shapelet = (shapelet - shapelet.mean()) / (shapelet.std() + 1e-6)
        candidates.append(shapelet.astype(np.float32))
    return candidates

def shapelet_distance_features(
    X: np.ndarray,
    shapelets: list[np.ndarray],
    batch_size: int = 64,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    rows = np.empty((len(X), len(shapelets)), dtype=np.float32)
    batch_size = max(1, int(batch_size))
    for col, shapelet in enumerate(shapelets):
        length = len(shapelet)
        if X.shape[1] < length:
            rows[:, col] = np.inf
            continue
        for start in range(0, len(X), batch_size):
            end = min(len(X), start + batch_size)
            windows = sliding_window_view(X[start:end], window_shape=length, axis=1)
            w_mean = windows.mean(axis=2, keepdims=True)
            w_std = windows.std(axis=2, keepdims=True) + 1e-6
            windows = (windows - w_mean) / w_std
            dist = np.sqrt(((windows - shapelet.reshape(1, 1, -1)) ** 2).mean(axis=2))
            rows[start:end, col] = dist.min(axis=1)
    rows[~np.isfinite(rows)] = np.nanmax(rows[np.isfinite(rows)]) if np.isfinite(rows).any() else 0.0
    return rows

def shapelet_features(dataset_name: str, X_train: np.ndarray, X_test: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Selects 32 TRAIN-only shapelet prototypes with lengths from 8, 16, 32, 64 that fit target_len.
    Z-normalizes and resamples rows.
    Returns: (train_features, test_features)
    """
    shapelet_target_len = min(512, max(8, int(target_len)))
    X_train_s = z_normalize(resample_rows(X_train, shapelet_target_len)).astype(np.float32)
    X_test_s = z_normalize(resample_rows(X_test, shapelet_target_len)).astype(np.float32)

    lengths = [length for length in [8, 16, 32, 64] if length <= shapelet_target_len]
    seed = 20260714 + stable_hash(dataset_name)
    shapelets = choose_shapelets(X_train_s, 32, lengths, seed)

    train_dist = shapelet_distance_features(X_train_s, shapelets)
    test_dist = shapelet_distance_features(X_test_s, shapelets)
    return train_dist, test_dist

def interval_slices(length: int, count: int) -> list[tuple[int, int]]:
    count = max(1, int(count))
    starts = np.linspace(0, length - 1, count, dtype=int)
    min_width = max(4, length // max(4, count))
    spans = []
    for idx, start in enumerate(starts):
        if idx == count - 1:
            end = length
        else:
            end = min(length, max(start + min_width, int(starts[idx + 1])))
        spans.append((int(start), int(max(start + 1, end))))
    spans.append((0, length))
    return spans

def summarize_segment(segment: np.ndarray) -> np.ndarray:
    segment = np.asarray(segment, dtype=np.float32)
    if segment.size == 0:
        return np.zeros(10, dtype=np.float32)
    qs = np.percentile(segment, [5, 25, 50, 75, 95]).astype(np.float32)
    x = np.linspace(-1.0, 1.0, len(segment), dtype=np.float32)
    slope = float(np.dot(segment - segment.mean(), x) / (np.dot(x, x) + 1e-9))
    return np.array(
        [
            float(segment.mean()),
            float(segment.std()),
            float(segment.min()),
            float(segment.max()),
            *qs.tolist(),
            slope,
        ],
        dtype=np.float32,
    )

def interval_features(X: np.ndarray) -> np.ndarray:
    """
    Z-normalized raw signal, first difference, and log-periodogram views.
    16 deterministic intervals plus one full-series interval per view.
    """
    X = np.asarray(X, dtype=np.float32)
    X_normalized = z_normalize(X)

    # 3 views: raw, diff, periodogram
    arrays = [X_normalized]
    # first difference
    diff_arr = np.diff(X_normalized, axis=1, prepend=X_normalized[:, :1]).astype(np.float32)
    arrays.append(diff_arr)
    # log-periodogram
    periodogram = np.log1p(np.abs(np.fft.rfft(X_normalized, axis=1))).astype(np.float32)
    if periodogram.shape[1] > 1:
        periodogram = periodogram[:, 1:]
    arrays.append(periodogram)

    rows = []
    for arr in arrays:
        spans = interval_slices(arr.shape[1], 16)
        for row_idx in range(arr.shape[0]):
            if len(rows) <= row_idx:
                rows.append([])
            for start, end in spans:
                rows[row_idx].append(summarize_segment(arr[row_idx, start:end]))

    return np.vstack([np.concatenate(parts) for parts in rows]).astype(np.float32)

def robust_score_pair(train_features: np.ndarray, test_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaling and center/spread fitting use TRAIN-normal features only.
    Score is the mean absolute robust deviation across all feature dimensions.
    """
    scaler = RobustScaler(quantile_range=(10, 90))
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    center = np.median(train_scaled, axis=0)
    spread = np.median(np.abs(train_scaled - center), axis=0) + 1e-6

    train_dev = np.abs(train_scaled - center) / spread
    test_dev = np.abs(test_scaled - center) / spread

    return train_dev.mean(axis=1), test_dev.mean(axis=1)

def source_config_hash() -> str:
    config = {
        "spectral_derivative": {
            "features": [
                "mean", "std", "median", "p90_p10", "max_abs",
                "bands_8", "centroid", "entropy", "peak"
            ],
            "fft_bands": 8,
            "normalization": "aligned_series_input",
        },
        "shapelet_normal_prototype": {
            "num_prototypes": 32,
            "lengths": [8, 16, 32, 64],
            "target_len_cap": 512,
            "sample_seed_base": 20260714,
            "seed_hash": "weighted_unicode_codepoint_sum",
            "normalization": "per_series_zscore",
        },
        "interval_quantile": {
            "intervals": 16,
            "include_diff": True,
            "include_periodogram": True,
            "metrics": ["mean", "std", "min", "max", "q05", "q25", "q50", "q75", "q95", "slope"],
            "normalization": "per_series_zscore",
        },
        "score": {
            "scaler": "RobustScaler",
            "quantile_range": [10, 90],
            "center": "train_scaled_median",
            "spread": "train_scaled_median_absolute_deviation_plus_1e-6",
            "aggregation": "mean_absolute_robust_deviation",
        },
        "alignment": {"source": "train_length_median", "min": 8, "max": 2048},
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode("utf-8")).hexdigest()
