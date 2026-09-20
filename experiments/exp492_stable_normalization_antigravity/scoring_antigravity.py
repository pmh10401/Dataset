"""Paired old/stable LocalGap scores with shared frozen Exp449 encoders."""
from __future__ import annotations

import numpy as np
import torch

from ..exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from ..exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
    _HydraMultiRocketRepresentationAntigravity,
    _features_antigravity,
    _rows_antigravity as _representation_rows_antigravity,
    make_representation_antigravity,
)
from ..exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    KNN_NEIGHBORS_ANTIGRAVITY,
    QUERY_CHUNK_SIZE_ANTIGRAVITY,
    _REQUESTED_KERNELS_ANTIGRAVITY,
    _feature_matrix_antigravity,
    _reference_indices_antigravity,
    _rows_antigravity,
    crossfit_plan_antigravity,
)
from .normalization_antigravity import StableLocalGapModelAntigravity, StableSparseScalerAntigravity

_METHODS_ANTIGRAVITY = frozenset({"hydra_local_gap", "hydra_multirocket_local_gap"})


def _jsonable_guard_antigravity(values: np.ndarray) -> list:
    array = np.asarray(values).reshape(-1)
    if array.dtype == bool or np.issubdtype(array.dtype, np.bool_):
        return [bool(item) for item in array.tolist()]
    floats = np.asarray(array, dtype=float)
    if not np.isfinite(floats).all():
        raise ValueError("non-finite diagnostics")
    return [float(item) for item in floats.tolist()]


def _abs_stats_antigravity(values: np.ndarray, field: str) -> tuple[float, int]:
    finite = np.isfinite(values)
    if values.size == 0 or not finite.all():
        raise ValueError(f"non-finite {field}")
    return float(np.max(np.abs(values))), int(finite.sum())


def _sparse_diagnostics_antigravity(scaler: StableSparseScalerAntigravity) -> dict:
    return {
        "sparse_original_spread_": _jsonable_guard_antigravity(scaler.original_spread_),
        "sparse_original_scale_": _jsonable_guard_antigravity(scaler.original_scale_),
        "sparse_tolerance_": _jsonable_guard_antigravity(scaler.tolerance_),
        "sparse_degeneracy_mask_": _jsonable_guard_antigravity(scaler.degeneracy_mask_),
        "sparse_changed_mask_": _jsonable_guard_antigravity(scaler.changed_mask_),
    }


def _robust_diagnostics_antigravity(scaler) -> dict:
    return {
        "robust_original_spread_": _jsonable_guard_antigravity(scaler.original_spread_),
        "robust_original_scale_": _jsonable_guard_antigravity(scaler.original_scale_),
        "robust_tolerance_": _jsonable_guard_antigravity(scaler.tolerance_),
        "robust_degeneracy_mask_": _jsonable_guard_antigravity(scaler.degeneracy_mask_),
        "robust_changed_mask_": _jsonable_guard_antigravity(scaler.changed_mask_),
    }


class _PairedHydraEncoderAntigravity:
    def __init__(self, seed: int) -> None:
        self._representation = make_representation_antigravity("hydra_local_gap", seed)
        self.stable_sparse = None
        self.dimension: int | None = None

    def fit_transform(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = _feature_matrix_antigravity(self._representation.fit_transform(rows))
        self.dimension = int(features.shape[1])
        return features, features

    def transform(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = _feature_matrix_antigravity(self._representation.transform(rows), expected_dimension=self.dimension)
        return features, features


class _PairedHybridEncoderAntigravity:
    def __init__(self, seed: int) -> None:
        self._upstream = _HydraMultiRocketRepresentationAntigravity(seed)
        self.stable_sparse = StableSparseScalerAntigravity()
        self.dimension: int | None = None

    def fit_transform(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        collection = _representation_rows_antigravity(rows)
        hydra_raw = self._upstream._hydra.fit_transform(collection)
        old_hydra = _features_antigravity(self._upstream._hydra_scaler.fit_transform(hydra_raw))
        stable_hydra = _features_antigravity(self.stable_sparse.fit_transform(hydra_raw))
        multirocket = _features_antigravity(
            self._upstream._multirocket_scaler.fit_transform(self._upstream._multirocket.fit_transform(collection))
        )
        old = _features_antigravity(np.concatenate((old_hydra, multirocket), axis=1))
        stable = _features_antigravity(np.concatenate((stable_hydra, multirocket), axis=1))
        if old.shape[1] != stable.shape[1]:
            raise ValueError("inconsistent feature dimensions")
        self.dimension = int(old.shape[1])
        return old, stable

    def transform(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        collection = _representation_rows_antigravity(rows)
        hydra_raw = self._upstream._hydra.transform(collection)
        old_hydra = _features_antigravity(self._upstream._hydra_scaler.transform(hydra_raw))
        stable_hydra = _features_antigravity(self.stable_sparse.transform(hydra_raw))
        multirocket = _features_antigravity(
            self._upstream._multirocket_scaler.transform(self._upstream._multirocket.transform(collection))
        )
        old = _features_antigravity(np.concatenate((old_hydra, multirocket), axis=1))
        stable = _features_antigravity(np.concatenate((stable_hydra, multirocket), axis=1))
        return (
            _feature_matrix_antigravity(old, expected_dimension=self.dimension),
            _feature_matrix_antigravity(stable, expected_dimension=self.dimension),
        )


def _make_encoder_antigravity(method: str, seed: int):
    if method == "hydra_multirocket_local_gap":
        return _PairedHybridEncoderAntigravity(seed)
    return _PairedHydraEncoderAntigravity(seed)


def _run_fold_antigravity(
    encoder,
    reference_rows: np.ndarray,
    query_rows: np.ndarray,
    *,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    old_features, stable_features = encoder.fit_transform(reference_rows)
    if old_features.shape[1] != stable_features.shape[1]:
        raise ValueError("inconsistent feature dimensions")
    old_model = LocalGapModelAntigravity(old_features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY)
    stable_model = StableLocalGapModelAntigravity(stable_features, neighbors=KNN_NEIGHBORS_ANTIGRAVITY)
    stats = {}
    def observe(old, stable):
        for key, values in (("old_feature", old), ("stable_feature", stable),
                            ("old_scaled", old_model.scaler.transform(old)),
                            ("stable_scaled", stable_model.scaler.transform(stable))):
            maximum, count = _abs_stats_antigravity(values, key)
            previous_max, previous_count = stats.get(key, (0.0, 0))
            stats[key] = (max(previous_max, maximum), previous_count + count)
    observe(old_features, stable_features)
    old_scores, stable_scores = [], []
    for start in range(0, len(query_rows), QUERY_CHUNK_SIZE_ANTIGRAVITY):
        chunk = query_rows[start : start + QUERY_CHUNK_SIZE_ANTIGRAVITY]
        old_query, stable_query = encoder.transform(chunk)
        observe(old_query, stable_query)
        old_scores.append(old_model.score(old_query))
        stable_scores.append(stable_model.score(stable_query))
    old_vector = np.concatenate(old_scores) if old_scores else np.asarray([], dtype=float)
    stable_vector = np.concatenate(stable_scores) if stable_scores else np.asarray([], dtype=float)
    if old_vector.shape != (len(query_rows),) or stable_vector.shape != (len(query_rows),):
        raise ValueError("score vectors must match query length")
    if not np.isfinite(old_vector).all() or not np.isfinite(stable_vector).all():
        raise ValueError("non-finite local-gap scores")
    diagnostics = {
        "fold": int(fold),
        **{key + "_max_abs": value[0] for key, value in stats.items()},
        **{key + "_finite_count": value[1] for key, value in stats.items()},
        "old_local_radius_scale": old_model.scale,
        "stable_local_radius_scale": stable_model.scale,
        **_robust_diagnostics_antigravity(stable_model.scaler),
    }
    if encoder.stable_sparse is not None:
        diagnostics.update(_sparse_diagnostics_antigravity(encoder.stable_sparse))
    return old_vector, stable_vector, diagnostics


def score_pair_antigravity(train_rows, test_rows, method, *, seed) -> dict:
    torch.set_num_threads(1)
    train = _rows_antigravity(train_rows, "TRAIN")
    test = _rows_antigravity(test_rows, "TEST")
    if train.shape[1] != test.shape[1]:
        raise ValueError("TRAIN and TEST row shape must match")
    if type(method) is not str or method not in _METHODS_ANTIGRAVITY:
        raise ValueError("method is not a fixed Exp492 source")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be an exact non-negative integer")
    requested_kernels = _REQUESTED_KERNELS_ANTIGRAVITY[method]
    old_train_oof = np.empty(len(train), dtype=float)
    stable_train_oof = np.empty(len(train), dtype=float)
    diagnostics = []
    dimensions: set[int] = set()
    for fold_index, (fit_indices, query_indices) in enumerate(crossfit_plan_antigravity(len(train), seed=seed)):
        reference_indices = _reference_indices_antigravity(
            fit_indices, seed=seed + 1009 * (fold_index + 1) + requested_kernels
        )
        encoder = _make_encoder_antigravity(method, seed)
        old_scores, stable_scores, fold_diagnostics = _run_fold_antigravity(
            encoder, train[reference_indices], train[query_indices], fold=fold_index
        )
        dimensions.add(int(encoder.dimension))
        old_train_oof[query_indices] = old_scores
        stable_train_oof[query_indices] = stable_scores
        diagnostics.append(fold_diagnostics)
    full_reference_indices = _reference_indices_antigravity(
        np.arange(len(train), dtype=int), seed=seed + 7919 + requested_kernels
    )
    encoder = _make_encoder_antigravity(method, seed)
    old_test, stable_test, full_diagnostics = _run_fold_antigravity(
        encoder, train[full_reference_indices], test, fold=5
    )
    dimensions.add(int(encoder.dimension))
    if len(dimensions) != 1:
        raise ValueError(f"inconsistent feature dimensions: {sorted(dimensions)}")
    if not np.isfinite(old_train_oof).all() or not np.isfinite(stable_train_oof).all():
        raise ValueError("non-finite local-gap scores")
    if old_test.shape != (len(test),) or stable_test.shape != (len(test),):
        raise ValueError("score vectors must match query length")
    diagnostics.append(full_diagnostics)
    if len(diagnostics) != 6:
        raise ValueError("expected six representation fits")
    return {
        "old": {"train_oof": old_train_oof, "test_scores": old_test},
        "stable": {"train_oof": stable_train_oof, "test_scores": stable_test},
        "feature_dimension": dimensions.pop(),
        "representation_fits": 6,
        "diagnostics": diagnostics,
    }
