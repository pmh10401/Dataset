"""Feature-compatible adapter for fixed- and variable-width TRAIN rows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def extract_ragged_routing_features_antigravity(
    train_rows: Sequence[np.ndarray],
) -> dict[str, float]:
    """Apply the frozen routing feature formulas without padding or resampling."""
    if len(train_rows) < 2:
        raise ValueError("train_rows must contain at least two rows")
    rows = [np.asarray(row, dtype=np.float64) for row in train_rows]
    if any(row.ndim != 1 or row.size < 2 or not np.isfinite(row).all() for row in rows):
        raise ValueError("each TRAIN row must be a finite 1-D array with at least two values")

    entropies: list[float] = []
    diff_ratios: list[float] = []
    zero_crossing_rates: list[float] = []
    for row in rows:
        centered = row - row.mean()
        fft_power = np.abs(np.fft.rfft(centered)) ** 2
        power_sum = float(fft_power.sum())
        safe_sum = power_sum if power_sum > 0.0 else 1.0
        probs = fft_power / safe_sum
        log_probs = np.where(probs > 0.0, np.log(probs + 1e-300), 0.0)
        entropy = -(probs * log_probs).sum() / np.log(max(2, probs.size))
        entropies.append(float(entropy))
        diff_ratios.append(float(np.diff(row).std() / (row.std() + 1e-12)))
        zero_crossing_rates.append(float(np.mean(np.diff(np.sign(centered)) != 0)))

    flat = np.concatenate(rows)
    mean_value = float(flat.mean())
    std_value = float(flat.std() + 1e-12)
    skewness = float(np.mean((flat - mean_value) ** 3) / std_value**3)
    kurtosis = float(np.mean((flat - mean_value) ** 4) / std_value**4 - 3.0)
    return {
        "width": float(np.median([row.size for row in rows])),
        "n_train": float(len(rows)),
        "spectral_entropy": float(np.median(entropies)),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "diff_std_ratio": float(np.median(diff_ratios)),
        "zero_crossing_rate": float(np.median(zero_crossing_rates)),
    }


__all__ = ["extract_ragged_routing_features_antigravity"]
