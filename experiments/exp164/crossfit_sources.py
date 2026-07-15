from __future__ import annotations
import numpy as np

from experiments.exp162.full_coverage_score_core import (
    resample_rows,
    z_normalize,
    spectral_features,
    shapelet_features,
    interval_features,
    robust_score_pair,
)

def score_fold(
    dataset_name: str,
    fit_raw: list[np.ndarray],
    query_raw: list[np.ndarray],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], int]:
    """
    Computes feature extraction and robust score pairing for a single fold.
    Returns dictionary mapping source name to (fit_scores, query_scores).
    """
    # 1. target length: median of fit_raw series lengths, clipped to [8, 2048]
    lengths = [len(x) for x in fit_raw]
    target_len = int(np.clip(np.median(lengths), 8, 2048))

    # 2. Resample rows (alignment)
    fit_aligned = np.vstack([
        resample_rows(series.reshape(1, -1), target_len)[0]
        for series in fit_raw
    ]).astype(np.float32)

    query_aligned = np.vstack([
        resample_rows(series.reshape(1, -1), target_len)[0]
        for series in query_raw
    ]).astype(np.float32)

    # 3. Spectral features
    fit_spectral = spectral_features(fit_aligned)
    query_spectral = spectral_features(query_aligned)
    fit_spec_scores, query_spec_scores = robust_score_pair(fit_spectral, query_spectral)

    # 4. Shapelet features
    fit_shapelet, query_shapelet = shapelet_features(dataset_name, fit_aligned, query_aligned, target_len)
    fit_shap_scores, query_shap_scores = robust_score_pair(fit_shapelet, query_shapelet)

    # 5. Interval features
    fit_interval = interval_features(fit_aligned)
    query_interval = interval_features(query_aligned)
    fit_int_scores, query_int_scores = robust_score_pair(fit_interval, query_interval)

    return ({
        "spectral": (fit_spec_scores.astype(np.float64), query_spec_scores.astype(np.float64)),
        "shapelet": (fit_shap_scores.astype(np.float64), query_shap_scores.astype(np.float64)),
        "interval": (fit_int_scores.astype(np.float64), query_int_scores.astype(np.float64)),
    }, target_len)

def audit_dataset(dataset_name: str, raw_series: list[np.ndarray]) -> dict | None:
    """
    Runs cross-fit audit over all folds for a single dataset.
    Returns results dict or None if abstained.
    """
    n_train = len(raw_series)
    if n_train < 5:
        # Complete abstention
        return None

    from experiments.exp164.train_normal_risk_core import fold_plan, higher_threshold, source_candidate_flags, agreement_flags
    plan = fold_plan(n_train, 20260715)

    sources = ["spectral", "shapelet", "interval"]
    alphas = (0.005, 0.01, 0.02, 0.05)

    results = {
        "n_train": n_train,
        "fold_method": "leave-one-out" if n_train < 20 else "5-fold",
        "pseudo_run_indices": np.arange(n_train, dtype=np.int32),
        "query_scores": {src: np.zeros(n_train, dtype=np.float64) for src in sources},
        "fold_indices": np.zeros(n_train, dtype=np.int32),
        "fit_counts": np.zeros(n_train, dtype=np.int32),
        "target_lengths": np.zeros(n_train, dtype=np.int32),
        "thresholds": {alpha: {src: np.zeros(n_train, dtype=np.float64) for src in sources} for alpha in alphas},
        "flags": {alpha: {src: np.zeros(n_train, dtype=bool) for src in sources} for alpha in alphas},
        "agreement_2of3": {alpha: np.zeros(n_train, dtype=bool) for alpha in alphas},
        "agreement_3of3": {alpha: np.zeros(n_train, dtype=bool) for alpha in alphas},
    }

    for fold_idx, (fit_idx, query_idx) in enumerate(plan):
        fit_raw = [raw_series[idx] for idx in fit_idx]
        query_raw = [raw_series[idx] for idx in query_idx]

        fold_scores, target_len = score_fold(dataset_name, fit_raw, query_raw)
        results["fold_indices"][query_idx] = fold_idx
        results["fit_counts"][query_idx] = len(fit_idx)
        results["target_lengths"][query_idx] = target_len

        for src in sources:
            fit_scores, query_scores = fold_scores[src]
            results["query_scores"][src][query_idx] = query_scores
            for alpha in alphas:
                t = higher_threshold(fit_scores, alpha)
                results["thresholds"][alpha][src][query_idx] = t
                results["flags"][alpha][src][query_idx] = source_candidate_flags(query_scores, t)

    for alpha in alphas:
        flags_list = [results["flags"][alpha][src] for src in sources]
        results["agreement_2of3"][alpha] = agreement_flags(flags_list, 2)
        results["agreement_3of3"][alpha] = agreement_flags(flags_list, 3)

    return results
