from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.exp206_210.core import SEED, fold_plan
from experiments.exp162.full_coverage_score_core import (
    spectral_features,
    choose_shapelets,
    shapelet_distance_features,
    interval_features,
    robust_score_pair,
    z_normalize,
)


def alternative_oof_config_hash() -> str:
    config = {
        "experiment_seed": SEED,
        "folds": 5,
        "sources": {
            "spectral": "exp162_spectral_features",
            "shapelet": {
                "features": "32_fit_fold_only_prototypes",
                "lengths": [8, 16, 32, 64],
                "seed": "experiment_seed_plus_fold_index",
            },
            "interval": "exp162_interval_features",
        },
        "score": "fit_fold_only_robust_score_pair",
        "test": "full_train_fit",
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def fit_score_source(
    source: str, x_fit: np.ndarray, x_query: np.ndarray, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits feature extraction and robust scoring on x_fit and scores both x_fit and x_query.
    Returns: (fit_scores, query_scores)
    """
    x_fit = np.asarray(x_fit, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)

    if source == "spectral":
        fit_feat = spectral_features(x_fit)
        query_feat = spectral_features(x_query)
        fit_scores, query_scores = robust_score_pair(fit_feat, query_feat)
        return fit_scores, query_scores

    elif source == "shapelet":
        target_len = x_fit.shape[1]
        X_train_s = z_normalize(x_fit).astype(np.float32)
        X_query_s = z_normalize(x_query).astype(np.float32)

        lengths = [length for length in [8, 16, 32, 64] if length <= target_len]
        shapelets = choose_shapelets(X_train_s, 32, lengths, seed)

        fit_feat = shapelet_distance_features(X_train_s, shapelets)
        query_feat = shapelet_distance_features(X_query_s, shapelets)

        fit_scores, query_scores = robust_score_pair(fit_feat, query_feat)
        return fit_scores, query_scores

    elif source == "interval":
        fit_feat = interval_features(x_fit)
        query_feat = interval_features(x_query)
        fit_scores, query_scores = robust_score_pair(fit_feat, query_feat)
        return fit_scores, query_scores

    else:
        raise ValueError(f"Unknown source: {source}")


def crossfit_alternative_sources(
    x_train: np.ndarray, x_test: np.ndarray, *, folds: int = 5, seed: int = SEED
) -> dict[str, np.ndarray]:
    """
    Rebuild the alternative sources with deterministic cross-fitting on TRAIN.
    Then fit on full TRAIN and score TEST.
    """
    n_train = len(x_train)
    plan = fold_plan(n_train, folds, seed)

    payload = {}
    for source in ["spectral", "shapelet", "interval"]:
        train_oof = np.zeros(n_train, dtype=np.float32)

        # 1. Fold-by-fold cross-fitting for OOF
        for fold_idx, (fit_idx, query_idx) in enumerate(plan):
            x_fit = x_train[fit_idx]
            x_query = x_train[query_idx]
            _, query_scores = fit_score_source(source, x_fit, x_query, seed=seed + fold_idx)
            train_oof[query_idx] = query_scores.astype(np.float32)

        # 2. Fit on full TRAIN and score TEST
        full_train_scores, test_scores = fit_score_source(
            source, x_train, x_test, seed=seed
        )

        payload[f"{source}_train"] = train_oof
        payload[f"{source}_train_full"] = full_train_scores.astype(np.float32)
        payload[f"{source}_test"] = test_scores.astype(np.float32)

    return payload
