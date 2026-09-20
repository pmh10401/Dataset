"""Normal-TRAIN-only 18D behavioral descriptors from frozen Exp449 LocalGap OOF scores."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
    LocalGapModelAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
    make_representation_antigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    QUERY_CHUNK_SIZE_ANTIGRAVITY,
    _REQUESTED_KERNELS_ANTIGRAVITY,
    _feature_matrix_antigravity,
    _reference_indices_antigravity,
    crossfit_plan_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
    SEED_ANTIGRAVITY,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    _standardize_rows_antigravity,
    lower_median_length_antigravity,
)
from experiments.exp488_catch22_train_descriptors_antigravity.features_antigravity import (
    load_training_rows,
)


BEHAVIOR_KEYS_ANTIGRAVITY = tuple(
    f"{method}_{suffix}"
    for method in METHODS_ANTIGRAVITY
    for suffix in ("tail_q95", "tail_q99", "ab_spearman")
) + tuple(
    f"spearman_{left}_{right}"
    for left, right in combinations(METHODS_ANTIGRAVITY, 2)
)


def b_reference_indices_antigravity(
    reference_indices: np.ndarray, *, seed: int, fold_index: int
) -> np.ndarray:
    values = np.asarray(reference_indices, dtype=int)
    nref = int(len(values))
    if nref < 2:
        raise ValueError("nref>=2 is required")
    size = min(nref, max(2, math.floor(0.8 * nref)))
    selected = np.random.default_rng(seed + 104729 + fold_index).choice(
        values, size=size, replace=False
    )
    return np.sort(selected)


def _score_chunks_antigravity(
    representation: object, model: LocalGapModelAntigravity, rows: np.ndarray, *, feature_dimension: int
) -> np.ndarray:
    parts = [
        model.score(
            _feature_matrix_antigravity(
                representation.transform(rows[start : start + QUERY_CHUNK_SIZE_ANTIGRAVITY]),
                expected_dimension=feature_dimension,
            )
        )
        for start in range(0, len(rows), QUERY_CHUNK_SIZE_ANTIGRAVITY)
    ]
    scores = np.concatenate(parts) if parts else np.asarray([], dtype=float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError("behavior scores must be finite and match the query count")
    return scores


def _tail_ratios_antigravity(scores: np.ndarray) -> tuple[float, float]:
    q25, q50, q75, q95, q99 = np.quantile(
        scores, [0.25, 0.50, 0.75, 0.95, 0.99], method="linear"
    )
    scale = float(q75 - q25 + 1e-9)
    return float((q95 - q50) / scale), float((q99 - q50) / scale)


def spearman_or_zero_antigravity(left: np.ndarray, right: np.ndarray) -> tuple[float, bool]:
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Spearman inputs must be finite")
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return 0.0, True
    correlation = spearmanr(left, right).correlation
    if correlation is None or not np.isfinite(correlation):
        return 0.0, True
    return float(correlation), False


def assemble_behavior_antigravity(
    oof_a: Mapping[str, np.ndarray],
    oof_b: Mapping[str, np.ndarray],
) -> tuple[tuple[float, ...], dict[str, object]]:
    if tuple(oof_a) != METHODS_ANTIGRAVITY or tuple(oof_b) != METHODS_ANTIGRAVITY:
        raise ValueError("OOF maps must use the four frozen methods in order")
    values: list[float] = []
    constant = 0
    for method in METHODS_ANTIGRAVITY:
        a_scores = np.asarray(oof_a[method], dtype=float)
        b_scores = np.asarray(oof_b[method], dtype=float)
        if a_scores.shape != b_scores.shape or a_scores.ndim != 1 or not len(a_scores):
            raise ValueError("A/B OOF vectors must be aligned and non-empty")
        tail_q95, tail_q99 = _tail_ratios_antigravity(a_scores)
        corr, was_constant = spearman_or_zero_antigravity(a_scores, b_scores)
        constant += int(was_constant)
        values.extend((tail_q95, tail_q99, corr))
    for left, right in combinations(METHODS_ANTIGRAVITY, 2):
        corr, was_constant = spearman_or_zero_antigravity(
            np.asarray(oof_a[left], dtype=float), np.asarray(oof_a[right], dtype=float)
        )
        constant += int(was_constant)
        values.append(corr)
    if len(values) != 18 or not all(math.isfinite(value) for value in values):
        raise ValueError("behavior vector must be a finite 18-vector")
    return tuple(values), {"constant_correlation_count": int(constant)}


def extract_behavior_antigravity(
    train_rows: np.ndarray, seed: int = SEED_ANTIGRAVITY
) -> dict[str, object]:
    """Return an 18-vector from TRAIN-only LocalGap OOF; never reads TEST or labels."""
    train = np.asarray(train_rows, dtype=np.float32)
    if train.ndim != 2 or len(train) == 0 or not np.isfinite(train).all():
        raise ValueError("TRAIN must be a non-empty finite two-dimensional array")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    oof_a = {method: np.empty(len(train), dtype=float) for method in METHODS_ANTIGRAVITY}
    oof_b = {method: np.empty(len(train), dtype=float) for method in METHODS_ANTIGRAVITY}
    b_sizes: list[int] = []
    representation_fits = 0
    for method in METHODS_ANTIGRAVITY:
        requested = _REQUESTED_KERNELS_ANTIGRAVITY[method]
        fold_b_sizes = []
        for fold_index, (fit_indices, query_indices) in enumerate(
            crossfit_plan_antigravity(len(train), seed=seed)
        ):
            reference_a = _reference_indices_antigravity(
                fit_indices, seed=seed + 1009 * (fold_index + 1) + requested
            )
            reference_b = b_reference_indices_antigravity(
                reference_a, seed=seed, fold_index=fold_index
            )
            if np.intersect1d(reference_a, query_indices).size:
                raise ValueError("A reference overlaps held-out query indices")
            if np.intersect1d(reference_b, query_indices).size:
                raise ValueError("B reference overlaps held-out query indices")
            if not np.isin(reference_b, reference_a).all():
                raise ValueError("B reference must be drawn only from A reference")
            representation_a = make_representation_antigravity(method, seed)
            features_a = _feature_matrix_antigravity(representation_a.fit_transform(train[reference_a]))
            representation_fits += 1
            model_a = LocalGapModelAntigravity(features_a, neighbors=3)
            oof_a[method][query_indices] = _score_chunks_antigravity(
                representation_a, model_a, train[query_indices], feature_dimension=features_a.shape[1]
            )
            representation_b = make_representation_antigravity(method, seed)
            features_b = _feature_matrix_antigravity(representation_b.fit_transform(train[reference_b]))
            representation_fits += 1
            model_b = LocalGapModelAntigravity(features_b, neighbors=3)
            oof_b[method][query_indices] = _score_chunks_antigravity(
                representation_b, model_b, train[query_indices], feature_dimension=features_b.shape[1]
            )
            fold_b_sizes.append(int(len(reference_b)))
        b_sizes.append(fold_b_sizes)
    values, assembled = assemble_behavior_antigravity(oof_a, oof_b)
    diagnostics = {
        **assembled,
        "n_train": int(len(train)),
        "n_folds": 5,
        "representation_fits": int(representation_fits),
        "fold_b_sizes": b_sizes,
        "extraction_split": "normal_TRAIN_only",
    }
    return {
        "values": values,
        "keys": BEHAVIOR_KEYS_ANTIGRAVITY,
        "diagnostics": diagnostics,
        "seed": seed,
    }


def prepare_normal_train_matrix_antigravity(database, dataset: str) -> tuple[np.ndarray, dict[str, object]]:
    """Finite-row TRAIN-normal load plus Exp479 lower-median standardization; never reads TEST."""
    rows, membership = load_training_rows(database, dataset)
    finite = [row for row in rows if np.isfinite(row).all()]
    if len(finite) < 5:
        raise ValueError("at least five finite TRAIN rows are required")
    target_length = lower_median_length_antigravity(tuple(int(row.size) for row in finite))
    train = _standardize_rows_antigravity(finite, target_length)
    return train, {
        "train_membership_sha256": membership,
        "total_rows": int(len(rows)),
        "finite_rows": int(len(finite)),
        "target_length": int(target_length),
    }


__all__ = [
    "BEHAVIOR_KEYS_ANTIGRAVITY",
    "assemble_behavior_antigravity",
    "b_reference_indices_antigravity",
    "extract_behavior_antigravity",
    "prepare_normal_train_matrix_antigravity",
    "spearman_or_zero_antigravity",
]
