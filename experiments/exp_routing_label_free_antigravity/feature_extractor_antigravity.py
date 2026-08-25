"""Label-free feature extraction for routing decisions (antigravity)."""

from __future__ import annotations

import numpy as np


def extract_routing_features_antigravity(
    train_rows: np.ndarray,
    *,
    reference_scores: np.ndarray | None = None,
) -> dict[str, float]:
    """Return label-free features describing the dataset shape and score spread.

    Parameters use only TRAIN majority rows; no labels are consulted.
    """
    if train_rows.ndim != 2 or train_rows.shape[0] < 2:
        raise ValueError("train_rows must be 2-D with at least 2 rows")
    width = int(train_rows.shape[1])
    n_train = int(train_rows.shape[0])

    # Spectral entropy (normalised 0..1): low = periodic, high = noisy.
    fft_power = np.abs(np.fft.rfft(train_rows - train_rows.mean(axis=1, keepdims=True), axis=1)) ** 2
    power_sum = fft_power.sum(axis=1)
    safe_sum = np.where(power_sum > 0, power_sum, 1.0)
    probs = fft_power / safe_sum[:, None]
    log_probs = np.where(probs > 0, np.log(probs + 1e-300), 0.0)
    entropies = -(probs * log_probs).sum(axis=1) / np.log(max(2, probs.shape[1]))
    spectral_entropy = float(np.median(entropies))

    flat = train_rows.ravel()
    mean_v, std_v = float(flat.mean()), float(flat.std() + 1e-12)
    skewness = float(((flat - mean_v) ** 3).mean() / std_v**3)
    kurtosis = float(((flat - mean_v) ** 4).mean() / std_v**4 - 3.0)

    ref_score_var = (
        float(np.var(reference_scores)) if reference_scores is not None else float("nan")
    )

    return {
        "width": width,
        "n_train": n_train,
        "spectral_entropy": spectral_entropy,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "ref_score_var": ref_score_var,
    }
