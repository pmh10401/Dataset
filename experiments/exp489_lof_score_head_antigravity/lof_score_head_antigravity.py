"""Paired MiniRocket LocalGap control and LOF heads with one shared encoder fit."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

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
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import SEED_ANTIGRAVITY


REPRESENTATION_ANTIGRAVITY = "minirocket_local_gap"
REQUESTED_KERNELS_ANTIGRAVITY = _REQUESTED_KERNELS_ANTIGRAVITY[REPRESENTATION_ANTIGRAVITY]


def _rows_antigravity(rows: object, field: str) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{field} must be a non-empty finite two-dimensional array")
    return values


def _fit_lof_antigravity(scaled_reference: np.ndarray) -> LocalOutlierFactor:
    nref = int(len(scaled_reference))
    if nref < 2:
        raise ValueError("nref>=2 is required")
    model = LocalOutlierFactor(
        n_neighbors=min(3, nref - 1),
        novelty=True,
        metric="minkowski",
        p=2,
        contamination="auto",
        n_jobs=1,
    )
    model.fit(scaled_reference)
    return model


def _fit_paired_heads_antigravity(representation: object, reference_rows: np.ndarray):
    reference_features = _feature_matrix_antigravity(representation.fit_transform(reference_rows))
    nref = int(reference_features.shape[0])
    if nref < 2:
        raise ValueError("nref>=2 is required")
    dimension = int(reference_features.shape[1])
    control = LocalGapModelAntigravity(reference_features, neighbors=3)
    scaled_reference = control.scaler.transform(reference_features)
    lof = _fit_lof_antigravity(scaled_reference)
    return control, lof, dimension, nref


def _score_paired_chunks_antigravity(
    representation: object,
    control: LocalGapModelAntigravity,
    lof: LocalOutlierFactor,
    rows: np.ndarray,
    *,
    feature_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) == 0:
        raise ValueError("query rows must be non-empty")
    control_parts = []
    lof_parts = []
    for start in range(0, len(rows), QUERY_CHUNK_SIZE_ANTIGRAVITY):
        chunk = rows[start : start + QUERY_CHUNK_SIZE_ANTIGRAVITY]
        features = _feature_matrix_antigravity(
            representation.transform(chunk), expected_dimension=feature_dimension
        )
        control_parts.append(control.score(features))
        scaled = control.scaler.transform(features)
        lof_parts.append(np.asarray(-lof.score_samples(scaled), dtype=float))
    control_scores = np.concatenate(control_parts)
    lof_scores = np.concatenate(lof_parts)
    if control_scores.shape != (len(rows),) or lof_scores.shape != (len(rows),):
        raise ValueError("paired score shapes must match the query count")
    if not np.isfinite(control_scores).all() or not np.isfinite(lof_scores).all():
        raise ValueError("paired scores must be finite")
    return control_scores, lof_scores


def score_heads_antigravity(
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    seed: int = SEED_ANTIGRAVITY,
) -> dict[str, object]:
    """Fit MiniRocket once per fold/full and score LocalGap and LOF on the same features."""
    train = _rows_antigravity(train_rows, "TRAIN")
    test = _rows_antigravity(test_rows, "TEST")
    if train.shape[1] != test.shape[1]:
        raise ValueError("TRAIN and TEST row shape must match")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    control_oof = np.empty(len(train), dtype=float)
    lof_oof = np.empty(len(train), dtype=float)
    dimensions: set[int] = set()
    representation_fits = 0
    for fold_index, (fit_indices, query_indices) in enumerate(crossfit_plan_antigravity(len(train), seed=seed)):
        reference_indices = _reference_indices_antigravity(
            fit_indices, seed=seed + 1009 * (fold_index + 1) + REQUESTED_KERNELS_ANTIGRAVITY
        )
        representation = make_representation_antigravity(REPRESENTATION_ANTIGRAVITY, seed)
        control, lof, dimension, _nref = _fit_paired_heads_antigravity(
            representation, train[reference_indices]
        )
        representation_fits += 1
        dimensions.add(dimension)
        control_oof[query_indices], lof_oof[query_indices] = _score_paired_chunks_antigravity(
            representation,
            control,
            lof,
            train[query_indices],
            feature_dimension=dimension,
        )
    full_reference_indices = _reference_indices_antigravity(
        np.arange(len(train), dtype=int), seed=seed + 7919 + REQUESTED_KERNELS_ANTIGRAVITY
    )
    representation = make_representation_antigravity(REPRESENTATION_ANTIGRAVITY, seed)
    control, lof, dimension, _nref = _fit_paired_heads_antigravity(
        representation, train[full_reference_indices]
    )
    representation_fits += 1
    dimensions.add(dimension)
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    control_test, lof_test = _score_paired_chunks_antigravity(
        representation, control, lof, test, feature_dimension=dimension
    )
    if not np.isfinite(control_oof).all() or not np.isfinite(lof_oof).all():
        raise ValueError("paired scores must be finite")
    return {
        "seed": seed,
        "representation": REPRESENTATION_ANTIGRAVITY,
        "feature_dimension": dimension,
        "representation_fits": representation_fits,
        "control": {"train_oof": control_oof, "test_scores": control_test},
        "lof": {"train_oof": lof_oof, "test_scores": lof_test},
    }


__all__ = [
    "REPRESENTATION_ANTIGRAVITY",
    "REQUESTED_KERNELS_ANTIGRAVITY",
    "score_heads_antigravity",
]
