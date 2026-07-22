from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from experiments.exp206_210.alternative_oof import fit_score_source
from experiments.exp206_210.core import SEED, empirical_upper_pvalues, fold_plan


SOURCE_NAMES = ("spectral", "shapelet", "interval")
FOLDS = 5


def fold_transport_config_hash() -> str:
    implementation_files = [
        Path(__file__),
        Path(inspect.getsourcefile(fit_score_source) or ""),
    ]
    payload = {
        "seed": SEED,
        "folds": FOLDS,
        "sources": SOURCE_NAMES,
        "calibration": "heldout_train_scores_from_same_fold_model",
        "query": "test_scores_from_same_fold_model",
        "aggregations": ("median", "conservative_max"),
        "implementation_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in implementation_files
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_fold_transport_artifact(
    x_train: np.ndarray, x_test: np.ndarray, *, seed: int = SEED
) -> dict[str, np.ndarray]:
    x_train = np.asarray(x_train, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    plan = fold_plan(len(x_train), FOLDS, seed)
    artifact: dict[str, np.ndarray] = {}
    for source in SOURCE_NAMES:
        oof = np.full(len(x_train), np.nan, dtype=np.float32)
        fold_test_pvalues = np.zeros((FOLDS, len(x_test)), dtype=np.float32)
        fold_test_exceedance_counts = np.zeros((FOLDS, len(x_test)), dtype=np.int32)
        fold_calibration_sizes = np.zeros(FOLDS, dtype=np.int32)
        for fold_index, (fit_indices, calibration_indices) in enumerate(plan):
            calibration_size = len(calibration_indices)
            fold_query = np.vstack([x_train[calibration_indices], x_test])
            _, query_scores = fit_score_source(
                source,
                x_train[fit_indices],
                fold_query,
                seed=seed + fold_index,
            )
            calibration_scores = query_scores[:calibration_size]
            test_scores = query_scores[calibration_size:]
            oof[calibration_indices] = calibration_scores.astype(np.float32)
            fold_test_pvalues[fold_index] = empirical_upper_pvalues(
                calibration_scores, test_scores
            ).astype(np.float32)
            ordered = np.sort(np.asarray(calibration_scores, dtype=float))
            fold_test_exceedance_counts[fold_index] = (
                len(ordered) - np.searchsorted(ordered, test_scores, side="left")
            ).astype(np.int32)
            fold_calibration_sizes[fold_index] = len(calibration_scores)
        full_train, full_test = fit_score_source(
            source, x_train, x_test, seed=seed
        )
        artifact[f"{source}_train_oof"] = oof
        artifact[f"{source}_train_full"] = np.asarray(full_train, dtype=np.float32)
        artifact[f"{source}_test_full"] = np.asarray(full_test, dtype=np.float32)
        artifact[f"{source}_fold_test_pvalues"] = fold_test_pvalues
        artifact[f"{source}_fold_test_exceedance_counts"] = fold_test_exceedance_counts
        artifact[f"{source}_fold_calibration_sizes"] = fold_calibration_sizes
    if not all(np.all(np.isfinite(value)) for value in artifact.values()):
        raise ValueError("non-finite fold transport artifact")
    return artifact
