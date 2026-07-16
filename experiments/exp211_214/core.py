from __future__ import annotations

from typing import Mapping

import numpy as np

from experiments.exp206_210.core import rankdata_average, robust_location_scale


STABILITY_CONTRACT = {
    "min_rank_corr": 0.90,
    "max_abs_median_shift": 0.25,
    "min_mad_ratio": 0.80,
    "max_mad_ratio": 1.25,
}


def aggregate_fold_pvalues(pvalues: np.ndarray, method: str) -> np.ndarray:
    matrix = np.asarray(pvalues, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("fold p-values must have shape (folds, queries)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("fold p-values must be finite")
    if method == "median":
        return np.median(matrix, axis=0)
    if method == "conservative_max":
        return np.max(matrix, axis=0)
    raise ValueError(f"unknown fold p-value aggregation: {method}")


def pooled_cross_conformal_pvalues(
    exceedance_counts: np.ndarray, calibration_sizes: np.ndarray
) -> np.ndarray:
    counts = np.asarray(exceedance_counts, dtype=float)
    sizes = np.asarray(calibration_sizes, dtype=float)
    if counts.ndim != 2 or sizes.ndim != 1 or counts.shape[0] != len(sizes):
        raise ValueError("cross-conformal counts and calibration sizes do not align")
    if np.any(counts < 0) or np.any(sizes <= 0) or np.any(counts > sizes[:, None]):
        raise ValueError("invalid cross-conformal exceedance counts")
    return (1.0 + np.sum(counts, axis=0)) / (1.0 + float(np.sum(sizes)))


def _ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    values = np.unique(np.concatenate([left, right]))
    if not len(values):
        return 0.0
    left_cdf = np.searchsorted(np.sort(left), values, side="right") / len(left)
    right_cdf = np.searchsorted(np.sort(right), values, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _wasserstein_quantile(left: np.ndarray, right: np.ndarray) -> float:
    quantiles = np.linspace(0.0, 1.0, 201)
    return float(np.mean(np.abs(np.quantile(left, quantiles) - np.quantile(right, quantiles))))


def transport_metrics(
    oof_train: np.ndarray, full_train: np.ndarray, unlabeled_test: np.ndarray
) -> dict[str, float]:
    oof = np.asarray(oof_train, dtype=float)
    full = np.asarray(full_train, dtype=float)
    test = np.asarray(unlabeled_test, dtype=float)
    if oof.ndim != 1 or full.ndim != 1 or test.ndim != 1:
        raise ValueError("transport vectors must be one-dimensional")
    if len(oof) != len(full):
        raise ValueError("OOF and full-fit TRAIN vectors must have equal length")
    if not len(oof) or not len(test) or not np.all(np.isfinite(np.concatenate([oof, full, test]))):
        raise ValueError("transport vectors must be non-empty and finite")
    location, scale = robust_location_scale(oof)
    oof_mad = float(np.median(np.abs(oof - np.median(oof))))
    full_mad = float(np.median(np.abs(full - np.median(full))))
    if oof_mad <= 1e-12:
        mad_ratio = 1.0 if full_mad <= 1e-12 else float("inf")
    else:
        mad_ratio = full_mad / oof_mad
    corr = float(np.corrcoef(rankdata_average(oof), rankdata_average(full))[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "oof_full_rank_corr": corr,
        "oof_full_median_shift_robust": abs(float(np.median(full)) - location) / scale,
        "oof_full_mad_ratio": float(mad_ratio),
        "oof_full_ks": _ks_distance(oof, full),
        "oof_full_wasserstein": _wasserstein_quantile(oof, full),
        "oof_test_ks": _ks_distance(oof, test),
        "full_test_ks": _ks_distance(full, test),
        "oof_test_wasserstein": _wasserstein_quantile(oof, test),
        "full_test_wasserstein": _wasserstein_quantile(full, test),
    }


def source_stability(metrics: Mapping[str, float]) -> bool:
    return bool(
        float(metrics["oof_full_rank_corr"]) >= STABILITY_CONTRACT["min_rank_corr"]
        and abs(float(metrics["oof_full_median_shift_robust"]))
        <= STABILITY_CONTRACT["max_abs_median_shift"]
        and STABILITY_CONTRACT["min_mad_ratio"]
        <= float(metrics["oof_full_mad_ratio"])
        <= STABILITY_CONTRACT["max_mad_ratio"]
    )
