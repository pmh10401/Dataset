"""Immutable C0/E11/C11 fold scoring with Numba-only new numeric paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numba
import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .predecessor_antigravity import score_c0_fold_antigravity
from .spectral_entropy_antigravity import score_spectral_entropy_numba_antigravity
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _owned_array_antigravity(values: object, name: str, *, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if values.dtype != dtype or values.shape != shape:
        raise ValueError(f"{name} has an invalid exact dtype or shape")
    if dtype == np.dtype(np.float64) and not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _owned_train_antigravity(values: object) -> np.ndarray:
    return _owned_array_antigravity(
        values, "train140", dtype=np.dtype(np.float64),
        shape=(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, FROZEN_CONFIG_ANTIGRAVITY.interpolated_width),
    )


def _owned_indices_antigravity(values: object, name: str, *, minimum: int) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.dtype != np.dtype(np.int64) or values.size < minimum:
        raise ValueError(f"{name} must be a nonempty exact int64 vector")
    if not np.all((values >= 0) & (values < FROZEN_CONFIG_ANTIGRAVITY.normal_rows)):
        raise ValueError(f"{name} contains out-of-range indices")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must contain unique indices")
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@numba.njit(cache=False, fastmath=False)
def _combine_c11_numba_antigravity(c0: np.ndarray, ranks: np.ndarray, confirmed: np.ndarray) -> np.ndarray:
    output = np.empty_like(c0)
    for index in range(c0.size):
        output[index] = ranks[index] if confirmed[index] and ranks[index] > c0[index] else c0[index]
    return output


@numba.njit(cache=False, fastmath=False)
def _rederive_rank_confirmation_numba_antigravity(
    fit_raw: np.ndarray, query_raw: np.ndarray, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(query_raw.size, dtype=np.float64)
    confirmed = np.empty(query_raw.size, dtype=np.bool_)
    for query_index in range(query_raw.size):
        raw = query_raw[query_index]
        count = 0
        for fit_index in range(fit_raw.size):
            if fit_raw[fit_index] <= raw:
                count += 1
        ranks[query_index] = (1.0 + count) / (fit_raw.size + 1.0)
        confirmed[query_index] = raw > 0.0 and raw > maximum
    return ranks, confirmed


@numba.njit(cache=False, fastmath=False)
def _fit_raw_maximum_numba_antigravity(fit_raw: np.ndarray) -> float:
    maximum = fit_raw[0]
    for index in range(1, fit_raw.size):
        if fit_raw[index] > maximum:
            maximum = fit_raw[index]
    return maximum


def _ordered_mapping_antigravity(values: object, keys: tuple[str, ...], name: str) -> Mapping[str, object]:
    if not isinstance(values, Mapping) or tuple(values) != keys:
        raise ValueError(f"{name} must preserve frozen key order")
    return values


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    seed: int
    fold_index: int
    methods: tuple[str, str, str]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    fit_raw_entropy: np.ndarray
    query_raw_entropy: Mapping[str, np.ndarray]
    fit_normal_max_raw_entropy: float
    confirmed: Mapping[str, np.ndarray]
    input_train140: np.ndarray
    input_fit_indices: np.ndarray
    input_query_indices: np.ndarray
    backend: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed differs from frozen Exp470 seeds")
        if type(self.fold_index) is not int or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds:
            raise ValueError("fold_index differs from frozen Exp470 folds")
        if self.methods != _METHODS_ANTIGRAVITY or self.backend != "numba":
            raise ValueError("methods/backend differ from frozen C0/E11/C11 Numba authority")
        train = _owned_train_antigravity(self.input_train140)
        fit = _owned_indices_antigravity(self.input_fit_indices, "fit_indices", minimum=2)
        query = _owned_indices_antigravity(self.input_query_indices, "query_indices", minimum=1)
        if np.intersect1d(fit, query).size:
            raise ValueError("fit/query indices must be disjoint")
        original_source = _ordered_mapping_antigravity(self.original, _METHODS_ANTIGRAVITY, "original")
        primary_source = _ordered_mapping_antigravity(self.primary, _METHODS_ANTIGRAVITY, "primary")
        original = {name: _owned_array_antigravity(original_source[name], f"original[{name}]", dtype=np.dtype(np.float64), shape=(query.size,)) for name in _METHODS_ANTIGRAVITY}
        primary: dict[str, Mapping[str, np.ndarray]] = {}
        for method in _METHODS_ANTIGRAVITY:
            values = _ordered_mapping_antigravity(primary_source[method], _CHANGE_TYPES_ANTIGRAVITY, f"primary[{method}]")
            primary[method] = MappingProxyType({name: _owned_array_antigravity(values[name], f"primary[{method}][{name}]", dtype=np.dtype(np.float64), shape=(query.size,)) for name in _CHANGE_TYPES_ANTIGRAVITY})
        fit_raw = _owned_array_antigravity(self.fit_raw_entropy, "fit_raw_entropy", dtype=np.dtype(np.float64), shape=(fit.size,))
        maximum = self.fit_normal_max_raw_entropy
        if type(maximum) is not float or not isfinite(maximum):
            raise ValueError("fit_normal_max_raw_entropy must be exact finite float")
        if maximum != _fit_raw_maximum_numba_antigravity(fit_raw):
            raise ValueError("fit_normal_max_raw_entropy must equal retained fit raw maximum")
        query_raw_source = _ordered_mapping_antigravity(self.query_raw_entropy, _SCOPES_ANTIGRAVITY, "query_raw_entropy")
        confirmed_source = _ordered_mapping_antigravity(self.confirmed, _SCOPES_ANTIGRAVITY, "confirmed")
        query_raw = {scope: _owned_array_antigravity(query_raw_source[scope], f"query_raw_entropy[{scope}]", dtype=np.dtype(np.float64), shape=(query.size,)) for scope in _SCOPES_ANTIGRAVITY}
        confirmed = {scope: _owned_array_antigravity(confirmed_source[scope], f"confirmed[{scope}]", dtype=np.dtype(np.bool_), shape=(query.size,)) for scope in _SCOPES_ANTIGRAVITY}
        for scope in _SCOPES_ANTIGRAVITY:
            ranks, expected_confirmed = _rederive_rank_confirmation_numba_antigravity(fit_raw, query_raw[scope], maximum)
            stored_rank = original["E11"] if scope == "normal" else primary["E11"][scope]
            stored_c0 = original["C0"] if scope == "normal" else primary["C0"][scope]
            stored_c11 = original["C11"] if scope == "normal" else primary["C11"][scope]
            expected_c11 = _combine_c11_numba_antigravity(stored_c0, ranks, expected_confirmed)
            if not np.array_equal(stored_rank, ranks) or not np.array_equal(confirmed[scope], expected_confirmed):
                raise ValueError("stored E11 rank or confirmation differs from retained raw entropy")
            fallback = np.logical_not(expected_confirmed) | (ranks <= stored_c0)
            if (
                stored_c11.tobytes(order="C") != expected_c11.tobytes(order="C")
                or stored_c11[fallback].tobytes(order="C") != stored_c0[fallback].tobytes(order="C")
            ):
                raise ValueError("stored C11 differs from byte-exact compiled C0/E11 rule")
        object.__setattr__(self, "original", MappingProxyType(original))
        object.__setattr__(self, "primary", MappingProxyType(primary))
        object.__setattr__(self, "fit_raw_entropy", fit_raw)
        object.__setattr__(self, "query_raw_entropy", MappingProxyType(query_raw))
        object.__setattr__(self, "confirmed", MappingProxyType(confirmed))
        object.__setattr__(self, "input_train140", train)
        object.__setattr__(self, "input_fit_indices", fit)
        object.__setattr__(self, "input_query_indices", query)


def _scope_rows_antigravity(train: np.ndarray, query: np.ndarray) -> Mapping[str, np.ndarray]:
    generated = tuple(primary_rows_antigravity(row) for row in train[query])
    return MappingProxyType({"normal": train[query], **{change: np.stack(tuple(row[change] for row in generated)) for change in _CHANGE_TYPES_ANTIGRAVITY}})


def score_fold_methods_antigravity(train140: object, fit_indices: object, query_indices: object, *, seed: int, fold_index: int, authority: object) -> FoldMethodScoresAntigravity:
    """Score one synthetic fold: exact C0 plus exact Numba E11 and compiled C11."""
    train = _owned_train_antigravity(train140)
    fit = _owned_indices_antigravity(fit_indices, "fit_indices", minimum=2)
    query = _owned_indices_antigravity(query_indices, "query_indices", minimum=1)
    if np.intersect1d(fit, query).size:
        raise ValueError("fit/query indices must be disjoint")
    c0 = score_c0_fold_antigravity(train, fit, query, seed=seed, fold_index=fold_index, authority=authority)
    scopes = _scope_rows_antigravity(train, query)
    combined = np.concatenate(tuple(scopes[scope] for scope in _SCOPES_ANTIGRAVITY), axis=0)
    e11 = score_spectral_entropy_numba_antigravity(train[fit], combined)
    if e11.backend != "numba":
        raise ValueError("E11 scorer backend must be exact numba")
    width = query.size
    raw = {scope: e11.query_raw_entropy[index * width:(index + 1) * width] for index, scope in enumerate(_SCOPES_ANTIGRAVITY)}
    ranks = {scope: e11.query_ranks[index * width:(index + 1) * width] for index, scope in enumerate(_SCOPES_ANTIGRAVITY)}
    confirmed = {scope: e11.confirmation_mask[index * width:(index + 1) * width] for index, scope in enumerate(_SCOPES_ANTIGRAVITY)}
    c11_normal = _combine_c11_numba_antigravity(c0.normal_scores, ranks["normal"], confirmed["normal"])
    c11_primary = {change: _combine_c11_numba_antigravity(c0.primary_scores[change], ranks[change], confirmed[change]) for change in _CHANGE_TYPES_ANTIGRAVITY}
    return FoldMethodScoresAntigravity(seed=seed, fold_index=fold_index, methods=_METHODS_ANTIGRAVITY, original={"C0": c0.normal_scores, "E11": ranks["normal"], "C11": c11_normal}, primary={"C0": c0.primary_scores, "E11": {change: ranks[change] for change in _CHANGE_TYPES_ANTIGRAVITY}, "C11": c11_primary}, fit_raw_entropy=e11.fit_raw_entropy, query_raw_entropy=raw, fit_normal_max_raw_entropy=e11.fit_normal_max_raw_entropy, confirmed=confirmed, input_train140=train, input_fit_indices=fit, input_query_indices=query, backend="numba")


__all__ = ("FoldMethodScoresAntigravity", "_combine_c11_numba_antigravity", "score_fold_methods_antigravity")
