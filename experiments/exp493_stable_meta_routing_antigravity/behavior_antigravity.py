"""Stable-normalization updates to frozen Exp490 18D TRAIN-only behavior descriptors."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Mapping

import numpy as np
import torch

from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
    LocalGapModelAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
    make_representation_antigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    KNN_NEIGHBORS_ANTIGRAVITY,
    _REQUESTED_KERNELS_ANTIGRAVITY,
    _feature_matrix_antigravity,
    _reference_indices_antigravity,
    _rows_antigravity,
    _score_chunks_antigravity,
    crossfit_plan_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import (
    METHODS_ANTIGRAVITY,
)
from experiments.exp490_behavior_descriptors_antigravity.behavior_descriptors_antigravity import (
    _tail_ratios_antigravity,
    b_reference_indices_antigravity,
    spearman_or_zero_antigravity,
)
from experiments.exp492_stable_normalization_antigravity.scoring_antigravity import (
    _make_encoder_antigravity,
    _run_fold_antigravity,
)


_AFFECTED_METHODS_ANTIGRAVITY = METHODS_ANTIGRAVITY[2:]
_MULTI_METHOD_ANTIGRAVITY = METHODS_ANTIGRAVITY[1]
_UNAFFECTED_PAIR_ANTIGRAVITY = (METHODS_ANTIGRAVITY[0], METHODS_ANTIGRAVITY[1])
_PAIR_OFFSET_ANTIGRAVITY = len(METHODS_ANTIGRAVITY) * 3
_PAIR_INDEX_ANTIGRAVITY = {
    pair: _PAIR_OFFSET_ANTIGRAVITY + offset
    for offset, pair in enumerate(combinations(METHODS_ANTIGRAVITY, 2))
}
_CACHE_RTOL_ANTIGRAVITY = 1e-10
_CACHE_ATOL_ANTIGRAVITY = 1e-10


def _mapping_antigravity(values: object, field: str) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{field} must be a method-to-vector mapping")
    return values


def _finite_18_antigravity(values: object, field: str) -> np.ndarray:
    cached = np.asarray(values, dtype=float)
    if cached.shape != (18,) or not np.isfinite(cached).all():
        raise ValueError(f"{field} must be a finite 18-vector")
    return cached


def _aligned_finite_vector_antigravity(
    values: object, *, length: int | None, field: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{field} must be a non-empty one-dimensional vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    if length is not None and array.shape != (length,):
        raise ValueError(f"{field} must be aligned")
    return array


def _require_method_antigravity(mapping: Mapping[str, object], method: str, field: str) -> object:
    if method not in mapping:
        raise ValueError(f"{field} is missing {method}")
    return mapping[method]


def _method_vectors_antigravity(
    mapping: Mapping[str, object],
    methods: tuple[str, ...],
    *,
    length: int | None,
    field: str,
    required: bool,
) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    for method in methods:
        if method not in mapping:
            if required:
                raise ValueError(f"{field} is missing {method}")
            continue
        vectors[method] = _aligned_finite_vector_antigravity(
            mapping[method], length=length, field=f"{field}[{method}]"
        )
    return vectors


def _slot_antigravity(method: str) -> int:
    return METHODS_ANTIGRAVITY.index(method) * 3


def _tails_ab_antigravity(a_scores: np.ndarray, b_scores: np.ndarray) -> tuple[float, float, float]:
    if a_scores.shape != b_scores.shape:
        raise ValueError("A/B OOF vectors must be aligned and non-empty")
    tail_q95, tail_q99 = _tail_ratios_antigravity(a_scores)
    corr, _ = spearman_or_zero_antigravity(a_scores, b_scores)
    return float(tail_q95), float(tail_q99), float(corr)


def _matches_cache_antigravity(actual: tuple[float, ...] | float, cached: np.ndarray | float, field: str) -> None:
    if not np.allclose(actual, cached, rtol=_CACHE_RTOL_ANTIGRAVITY, atol=_CACHE_ATOL_ANTIGRAVITY):
        raise ValueError(f"reconstructed old {field} do not match cached old_values")


def _a_changed_antigravity(
    method: str, old_a: Mapping[str, np.ndarray], stable_a: Mapping[str, np.ndarray]
) -> bool:
    if method not in old_a or method not in stable_a:
        return False
    return not np.array_equal(old_a[method], stable_a[method])


def _available_a_antigravity(
    method: str, old_a: Mapping[str, np.ndarray], stable_a: Mapping[str, np.ndarray]
) -> np.ndarray | None:
    if method in stable_a:
        return stable_a[method]
    if method in old_a:
        return old_a[method]
    return None


def update_descriptors_antigravity(old_values, old_a, stable_a, old_b, stable_b) -> tuple[float, ...]:
    cached = _finite_18_antigravity(old_values, "old_values")
    old_a_map = _mapping_antigravity(old_a, "old_a")
    stable_a_map = _mapping_antigravity(stable_a, "stable_a")
    old_b_map = _mapping_antigravity(old_b, "old_b")
    stable_b_map = _mapping_antigravity(stable_b, "stable_b")
    length = None
    old_a_vectors: dict[str, np.ndarray] = {}
    stable_a_vectors: dict[str, np.ndarray] = {}
    old_b_vectors: dict[str, np.ndarray] = {}
    stable_b_vectors: dict[str, np.ndarray] = {}
    for method in _AFFECTED_METHODS_ANTIGRAVITY:
        old_a_vectors[method] = _aligned_finite_vector_antigravity(
            _require_method_antigravity(old_a_map, method, "old_a"),
            length=length,
            field=f"old_a[{method}]",
        )
        if length is None:
            length = int(len(old_a_vectors[method]))
        stable_a_vectors[method] = _aligned_finite_vector_antigravity(
            _require_method_antigravity(stable_a_map, method, "stable_a"),
            length=length,
            field=f"stable_a[{method}]",
        )
        old_b_vectors[method] = _aligned_finite_vector_antigravity(
            _require_method_antigravity(old_b_map, method, "old_b"),
            length=length,
            field=f"old_b[{method}]",
        )
        stable_b_vectors[method] = _aligned_finite_vector_antigravity(
            _require_method_antigravity(stable_b_map, method, "stable_b"),
            length=length,
            field=f"stable_b[{method}]",
        )
    old_a_vectors.update(
        _method_vectors_antigravity(
            old_a_map, METHODS_ANTIGRAVITY[:2], length=length, field="old_a", required=False
        )
    )
    stable_a_vectors.update(
        _method_vectors_antigravity(
            stable_a_map, METHODS_ANTIGRAVITY[:2], length=length, field="stable_a", required=False
        )
    )
    for method in METHODS_ANTIGRAVITY[:2]:
        if method in old_a_vectors and method in stable_a_vectors:
            if not np.array_equal(old_a_vectors[method], stable_a_vectors[method]):
                raise ValueError("unaffected A source changed")
    if METHODS_ANTIGRAVITY[0] not in old_a_vectors or METHODS_ANTIGRAVITY[0] not in stable_a_vectors:
        raise ValueError("MiniRocket A is required")
    updated = [float(value) for value in cached]
    for method in _AFFECTED_METHODS_ANTIGRAVITY:
        slot = _slot_antigravity(method)
        reconstructed = _tails_ab_antigravity(old_a_vectors[method], old_b_vectors[method])
        _matches_cache_antigravity(
            reconstructed, cached[slot : slot + 3], f"{method} tails/AB"
        )
        # Identical A can still have a changed B; always refresh affected tails and AB.
        replacement = _tails_ab_antigravity(stable_a_vectors[method], stable_b_vectors[method])
        updated[slot] = replacement[0]
        updated[slot + 1] = replacement[1]
        updated[slot + 2] = replacement[2]
    for left, right in combinations(METHODS_ANTIGRAVITY, 2):
        if (left, right) == _UNAFFECTED_PAIR_ANTIGRAVITY:
            continue
        slot = _PAIR_INDEX_ANTIGRAVITY[(left, right)]
        if left in old_a_vectors and right in old_a_vectors:
            old_corr, _ = spearman_or_zero_antigravity(old_a_vectors[left], old_a_vectors[right])
            _matches_cache_antigravity(float(old_corr), cached[slot], f"pairwise {left}/{right}")
        if not (
            _a_changed_antigravity(left, old_a_vectors, stable_a_vectors)
            or _a_changed_antigravity(right, old_a_vectors, stable_a_vectors)
        ):
            continue
        new_left = _available_a_antigravity(left, old_a_vectors, stable_a_vectors)
        new_right = _available_a_antigravity(right, old_a_vectors, stable_a_vectors)
        if new_left is None or new_right is None:
            raise ValueError("both A vectors are required for a changed pairwise correlation")
        if new_left.shape != new_right.shape:
            raise ValueError("pairwise A vectors must be aligned")
        new_corr, _ = spearman_or_zero_antigravity(new_left, new_right)
        updated[slot] = float(new_corr)
    if len(updated) != 18 or not all(math.isfinite(value) for value in updated):
        raise ValueError("behavior vector must be a finite 18-vector")
    return tuple(updated)


def _guard_train_antigravity(train_rows: object) -> np.ndarray:
    train = np.asarray(train_rows, dtype=np.float32)
    if train.ndim != 2 or len(train) == 0 or not np.isfinite(train).all():
        raise ValueError("TRAIN must be a non-empty finite two-dimensional array")
    return train


def _oof_multirocket_a_antigravity(train: np.ndarray, *, seed: int) -> tuple[np.ndarray, int]:
    method = _MULTI_METHOD_ANTIGRAVITY
    requested = _REQUESTED_KERNELS_ANTIGRAVITY[method]
    oof = np.empty(len(train), dtype=float)
    representation_fits = 0
    dimensions: set[int] = set()
    for fold_index, (fit_indices, query_indices) in enumerate(
        crossfit_plan_antigravity(len(train), seed=seed)
    ):
        reference_a = _reference_indices_antigravity(
            fit_indices, seed=seed + 1009 * (fold_index + 1) + requested
        )
        if np.intersect1d(reference_a, query_indices).size:
            raise ValueError("A reference overlaps held-out query indices")
        representation = make_representation_antigravity(method, seed)
        features = _feature_matrix_antigravity(representation.fit_transform(train[reference_a]))
        representation_fits += 1
        dimensions.add(int(features.shape[1]))
        oof[query_indices] = _score_chunks_antigravity(
            representation,
            LocalGapModelAntigravity(features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY),
            train[query_indices],
            feature_dimension=features.shape[1],
        )
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    if oof.shape != (len(train),) or not np.isfinite(oof).all():
        raise ValueError("multirocket A OOF must be a finite vector matching TRAIN length")
    return oof, representation_fits


def extract_behavior_antigravity(train, old_a, stable_a, old_values, *, seed) -> dict[str, object]:
    torch.set_num_threads(1)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    train_rows = _guard_train_antigravity(train)
    _finite_18_antigravity(old_values, "old_values")
    old_a_map = _mapping_antigravity(old_a, "old_a")
    stable_a_map = _mapping_antigravity(stable_a, "stable_a")
    n_train = int(len(train_rows))
    copied_old_a = dict(old_a_map)
    copied_stable_a = dict(stable_a_map)
    a_changed = False
    max_abs_a_change = 0.0
    for method in _AFFECTED_METHODS_ANTIGRAVITY:
        old_vector = _aligned_finite_vector_antigravity(
            _require_method_antigravity(copied_old_a, method, "old_a"),
            length=n_train,
            field=f"old_a[{method}]",
        )
        stable_vector = _aligned_finite_vector_antigravity(
            _require_method_antigravity(copied_stable_a, method, "stable_a"),
            length=n_train,
            field=f"stable_a[{method}]",
        )
        copied_old_a[method] = old_vector
        copied_stable_a[method] = stable_vector
        if not np.array_equal(old_vector, stable_vector):
            a_changed = True
        max_abs_a_change = max(max_abs_a_change, float(np.max(np.abs(old_vector - stable_vector))))
    extra_a = METHODS_ANTIGRAVITY[:2]
    copied_old_a.update(
        _method_vectors_antigravity(
            copied_old_a, extra_a, length=n_train, field="old_a", required=False
        )
    )
    copied_stable_a.update(
        _method_vectors_antigravity(
            copied_stable_a, extra_a, length=n_train, field="stable_a", required=False
        )
    )
    old_b: dict[str, np.ndarray] = {}
    stable_b: dict[str, np.ndarray] = {}
    arrays: dict[str, np.ndarray] = {}
    representation_fits = 0
    for method in _AFFECTED_METHODS_ANTIGRAVITY:
        requested = _REQUESTED_KERNELS_ANTIGRAVITY[method]
        old_b[method] = np.empty(n_train, dtype=float)
        stable_b[method] = np.empty(n_train, dtype=float)
        dimensions: set[int] = set()
        for fold_index, (fit_indices, query_indices) in enumerate(
            crossfit_plan_antigravity(n_train, seed=seed)
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
            encoder = _make_encoder_antigravity(method, seed)
            old_scores, stable_scores, _fold_diagnostics = _run_fold_antigravity(
                encoder, train_rows[reference_b], train_rows[query_indices], fold=fold_index
            )
            representation_fits += 1
            dimensions.add(int(encoder.dimension))
            old_b[method][query_indices] = old_scores
            stable_b[method][query_indices] = stable_scores
        if len(dimensions) != 1:
            raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
        if not np.isfinite(old_b[method]).all() or not np.isfinite(stable_b[method]).all():
            raise ValueError("paired B scores must be finite")
        arrays[f"old_b__{method}"] = old_b[method]
        arrays[f"stable_b__{method}"] = stable_b[method]
    if a_changed:
        multi_a, multi_fits = _oof_multirocket_a_antigravity(train_rows, seed=seed)
        representation_fits += multi_fits
        copied_old_a[_MULTI_METHOD_ANTIGRAVITY] = multi_a
        copied_stable_a[_MULTI_METHOD_ANTIGRAVITY] = multi_a
        arrays["multirocket_a"] = multi_a
    values = update_descriptors_antigravity(
        old_values, copied_old_a, copied_stable_a, old_b, stable_b
    )
    return {
        "values": values,
        "arrays": arrays,
        "diagnostics": {
            "representation_fits": int(representation_fits),
            "a_changed": bool(a_changed),
            "max_abs_a_change": float(max_abs_a_change),
            "n_train": n_train,
        },
    }


def score_multirocket_test_antigravity(train, test, *, seed) -> np.ndarray:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    train_rows = _rows_antigravity(train, "TRAIN")
    test_rows = _rows_antigravity(test, "TEST")
    if train_rows.shape[1] != test_rows.shape[1]:
        raise ValueError("TRAIN and TEST row shape must match")
    requested = _REQUESTED_KERNELS_ANTIGRAVITY[_MULTI_METHOD_ANTIGRAVITY]
    full_reference_indices = _reference_indices_antigravity(
        np.arange(len(train_rows), dtype=int), seed=seed + 7919 + requested
    )
    representation = make_representation_antigravity(_MULTI_METHOD_ANTIGRAVITY, seed)
    full_features = _feature_matrix_antigravity(
        representation.fit_transform(train_rows[full_reference_indices])
    )
    scores = _score_chunks_antigravity(
        representation,
        LocalGapModelAntigravity(full_features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY),
        test_rows,
        feature_dimension=full_features.shape[1],
    )
    if scores.shape != (len(test_rows),) or not np.isfinite(scores).all():
        raise ValueError("TEST scores must be a finite vector matching TEST length")
    return scores


__all__ = [
    "extract_behavior_antigravity",
    "score_multirocket_test_antigravity",
    "update_descriptors_antigravity",
]
