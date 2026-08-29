"""Immutable C0/E12/C12 fold scoring with retained ACF peak-lag evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numba
import numpy as np

from .acf_peak_lag_antigravity import (
    assemble_raw_rank_numba_antigravity,
    score_acf_peak_lag_numba_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .predecessor_antigravity import score_c0_fold_antigravity
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.method_order
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _owned_array_antigravity(
    values: object,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
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
        values,
        "train140",
        dtype=np.dtype(np.float64),
        shape=(
            FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
            FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
        ),
    )


def _owned_indices_antigravity(values: object, name: str, *, minimum: int) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.ndim != 1
        or values.dtype != np.dtype(np.int64)
        or values.size < minimum
    ):
        raise ValueError(f"{name} must be an exact int64 vector")
    if not np.all((values >= 0) & (values < FROZEN_CONFIG_ANTIGRAVITY.normal_rows)):
        raise ValueError(f"{name} contains out-of-range indices")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must contain unique indices")
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _ordered_mapping_antigravity(
    values: object,
    keys: tuple[str, ...],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(values, Mapping) or tuple(values) != keys:
        raise ValueError(f"{name} must preserve frozen key order")
    return values


def _owned_scope_arrays_antigravity(
    values: object,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> MappingProxyType:
    source = _ordered_mapping_antigravity(values, _SCOPES_ANTIGRAVITY, name)
    return MappingProxyType(
        {
            scope: _owned_array_antigravity(
                source[scope], f"{name}[{scope}]", dtype=dtype, shape=shape
            )
            for scope in _SCOPES_ANTIGRAVITY
        }
    )


@numba.njit(cache=False, fastmath=False)
def _combine_c12_numba_antigravity(
    c0: np.ndarray,
    ranks: np.ndarray,
    confirmed: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(c0)
    for index in range(c0.size):
        output[index] = (
            ranks[index]
            if confirmed[index] and ranks[index] > c0[index]
            else c0[index]
        )
    return output


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    seed: int
    fold_index: int
    methods: tuple[str, str, str]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    fit_early_peak_lags: np.ndarray
    fit_late_peak_lags: np.ndarray
    fit_peak_presence: np.ndarray
    fit_shifts: np.ndarray
    fit_raw_shift: np.ndarray
    query_early_peak_lags: Mapping[str, np.ndarray]
    query_late_peak_lags: Mapping[str, np.ndarray]
    query_peak_presence: Mapping[str, np.ndarray]
    query_shifts: Mapping[str, np.ndarray]
    query_raw_shift: Mapping[str, np.ndarray]
    fit_normal_max_raw_shift: float
    confirmed: Mapping[str, np.ndarray]
    input_train140: np.ndarray
    input_fit_indices: np.ndarray
    input_query_indices: np.ndarray
    backend: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed differs from frozen Exp471 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index differs from frozen Exp471 folds")
        if self.methods != _METHODS_ANTIGRAVITY or self.backend != "numba":
            raise ValueError("methods/backend differ from frozen C0/E12/C12 authority")

        train = _owned_train_antigravity(self.input_train140)
        fit = _owned_indices_antigravity(self.input_fit_indices, "fit_indices", minimum=2)
        query = _owned_indices_antigravity(self.input_query_indices, "query_indices", minimum=1)
        if np.intersect1d(fit, query).size:
            raise ValueError("fit/query indices must be disjoint")

        original_source = _ordered_mapping_antigravity(self.original, _METHODS_ANTIGRAVITY, "original")
        primary_source = _ordered_mapping_antigravity(self.primary, _METHODS_ANTIGRAVITY, "primary")
        original = {
            method: _owned_array_antigravity(
                original_source[method],
                f"original[{method}]",
                dtype=np.dtype(np.float64),
                shape=(query.size,),
            )
            for method in _METHODS_ANTIGRAVITY
        }
        primary: dict[str, Mapping[str, np.ndarray]] = {}
        for method in _METHODS_ANTIGRAVITY:
            values = _ordered_mapping_antigravity(
                primary_source[method], _CHANGE_TYPES_ANTIGRAVITY, f"primary[{method}]"
            )
            primary[method] = MappingProxyType(
                {
                    change: _owned_array_antigravity(
                        values[change],
                        f"primary[{method}][{change}]",
                        dtype=np.dtype(np.float64),
                        shape=(query.size,),
                    )
                    for change in _CHANGE_TYPES_ANTIGRAVITY
                }
            )

        fit_early = _owned_array_antigravity(
            self.fit_early_peak_lags, "fit_early_peak_lags", dtype=np.dtype(np.int64), shape=(fit.size,)
        )
        fit_late = _owned_array_antigravity(
            self.fit_late_peak_lags, "fit_late_peak_lags", dtype=np.dtype(np.int64), shape=(fit.size,)
        )
        fit_presence = _owned_array_antigravity(
            self.fit_peak_presence, "fit_peak_presence", dtype=np.dtype(np.bool_), shape=(fit.size, 2)
        )
        fit_shifts = _owned_array_antigravity(
            self.fit_shifts, "fit_shifts", dtype=np.dtype(np.int64), shape=(fit.size,)
        )
        fit_raw = _owned_array_antigravity(
            self.fit_raw_shift, "fit_raw_shift", dtype=np.dtype(np.float64), shape=(fit.size,)
        )
        query_early = _owned_scope_arrays_antigravity(
            self.query_early_peak_lags, "query_early_peak_lags", dtype=np.dtype(np.int64), shape=(query.size,)
        )
        query_late = _owned_scope_arrays_antigravity(
            self.query_late_peak_lags, "query_late_peak_lags", dtype=np.dtype(np.int64), shape=(query.size,)
        )
        query_presence = _owned_scope_arrays_antigravity(
            self.query_peak_presence, "query_peak_presence", dtype=np.dtype(np.bool_), shape=(query.size, 2)
        )
        query_shifts = _owned_scope_arrays_antigravity(
            self.query_shifts, "query_shifts", dtype=np.dtype(np.int64), shape=(query.size,)
        )
        query_raw = _owned_scope_arrays_antigravity(
            self.query_raw_shift, "query_raw_shift", dtype=np.dtype(np.float64), shape=(query.size,)
        )
        confirmed = _owned_scope_arrays_antigravity(
            self.confirmed, "confirmed", dtype=np.dtype(np.bool_), shape=(query.size,)
        )
        maximum = self.fit_normal_max_raw_shift
        if type(maximum) is not float or not isfinite(maximum):
            raise ValueError("fit_normal_max_raw_shift must be an exact finite float")

        if not np.array_equal(fit_shifts, fit_late - fit_early):
            raise ValueError("fit shift differs from retained peak lags")
        if not np.array_equal(fit_presence[:, 0], fit_early != 0) or not np.array_equal(
            fit_presence[:, 1], fit_late != 0
        ):
            raise ValueError("fit peak presence differs from retained peak lags")
        for scope in _SCOPES_ANTIGRAVITY:
            if not np.array_equal(query_shifts[scope], query_late[scope] - query_early[scope]):
                raise ValueError(f"query shift differs from retained peak lags: {scope}")
            if not np.array_equal(query_presence[scope][:, 0], query_early[scope] != 0) or not np.array_equal(
                query_presence[scope][:, 1], query_late[scope] != 0
            ):
                raise ValueError(f"query peak presence differs from retained peak lags: {scope}")
            expected_fit_raw, expected_query_raw, expected_ranks, expected_confirmed, expected_maximum = (
                assemble_raw_rank_numba_antigravity(
                    fit_shifts.astype(np.float64), query_shifts[scope].astype(np.float64)
                )
            )
            stored_rank = original["E12"] if scope == "normal" else primary["E12"][scope]
            stored_c0 = original["C0"] if scope == "normal" else primary["C0"][scope]
            stored_c12 = original["C12"] if scope == "normal" else primary["C12"][scope]
            expected_c12 = _combine_c12_numba_antigravity(stored_c0, expected_ranks, expected_confirmed)
            if not np.array_equal(fit_raw, expected_fit_raw) or maximum != float(expected_maximum):
                raise ValueError("retained fit raw shift or maximum differs from peak shifts")
            if (
                not np.array_equal(query_raw[scope], expected_query_raw)
                or not np.array_equal(stored_rank, expected_ranks)
                or not np.array_equal(confirmed[scope], expected_confirmed)
            ):
                raise ValueError("stored E12 raw, rank, or confirmation differs from peak shifts")
            fallback = np.logical_not(expected_confirmed) | (expected_ranks <= stored_c0)
            if (
                stored_c12.tobytes(order="C") != expected_c12.tobytes(order="C")
                or stored_c12[fallback].tobytes(order="C") != stored_c0[fallback].tobytes(order="C")
            ):
                raise ValueError("stored C12 differs from byte-exact compiled C0/E12 rule")

        object.__setattr__(self, "original", MappingProxyType(original))
        object.__setattr__(self, "primary", MappingProxyType(primary))
        object.__setattr__(self, "fit_early_peak_lags", fit_early)
        object.__setattr__(self, "fit_late_peak_lags", fit_late)
        object.__setattr__(self, "fit_peak_presence", fit_presence)
        object.__setattr__(self, "fit_shifts", fit_shifts)
        object.__setattr__(self, "fit_raw_shift", fit_raw)
        object.__setattr__(self, "query_early_peak_lags", query_early)
        object.__setattr__(self, "query_late_peak_lags", query_late)
        object.__setattr__(self, "query_peak_presence", query_presence)
        object.__setattr__(self, "query_shifts", query_shifts)
        object.__setattr__(self, "query_raw_shift", query_raw)
        object.__setattr__(self, "confirmed", confirmed)
        object.__setattr__(self, "input_train140", train)
        object.__setattr__(self, "input_fit_indices", fit)
        object.__setattr__(self, "input_query_indices", query)


def _scope_rows_antigravity(train: np.ndarray, query: np.ndarray) -> Mapping[str, np.ndarray]:
    generated = tuple(primary_rows_antigravity(row) for row in train[query])
    return MappingProxyType(
        {
            "normal": train[query],
            **{
                change: np.stack(tuple(row[change] for row in generated))
                for change in _CHANGE_TYPES_ANTIGRAVITY
            },
        }
    )


def _split_scopes_antigravity(values: np.ndarray, width: int) -> dict[str, np.ndarray]:
    return {
        scope: values[index * width : (index + 1) * width]
        for index, scope in enumerate(_SCOPES_ANTIGRAVITY)
    }


def score_fold_methods_antigravity(
    train140: object,
    fit_indices: object,
    query_indices: object,
    *,
    seed: int,
    fold_index: int,
    authority: object,
) -> FoldMethodScoresAntigravity:
    """Score one synthetic fold without mixing query rows into fit evidence."""
    train = _owned_train_antigravity(train140)
    fit = _owned_indices_antigravity(fit_indices, "fit_indices", minimum=2)
    query = _owned_indices_antigravity(query_indices, "query_indices", minimum=1)
    if np.intersect1d(fit, query).size:
        raise ValueError("fit/query indices must be disjoint")
    c0 = score_c0_fold_antigravity(
        train, fit, query, seed=seed, fold_index=fold_index, authority=authority
    )
    scopes = _scope_rows_antigravity(train, query)
    combined = np.concatenate(tuple(scopes[scope] for scope in _SCOPES_ANTIGRAVITY), axis=0)
    e12 = score_acf_peak_lag_numba_antigravity(train[fit], combined)
    if e12.backend != "numba":
        raise ValueError("E12 scorer backend must be exact numba")
    width = query.size
    early = _split_scopes_antigravity(e12.query_early_peak_lags, width)
    late = _split_scopes_antigravity(e12.query_late_peak_lags, width)
    presence = _split_scopes_antigravity(e12.query_peak_presence, width)
    shifts = _split_scopes_antigravity(e12.query_shifts, width)
    raw = _split_scopes_antigravity(e12.query_raw_shift, width)
    ranks = _split_scopes_antigravity(e12.query_ranks, width)
    confirmed = _split_scopes_antigravity(e12.confirmation_mask, width)
    c12_normal = _combine_c12_numba_antigravity(
        c0.normal_scores, ranks["normal"], confirmed["normal"]
    )
    c12_primary = {
        change: _combine_c12_numba_antigravity(
            c0.primary_scores[change], ranks[change], confirmed[change]
        )
        for change in _CHANGE_TYPES_ANTIGRAVITY
    }
    return FoldMethodScoresAntigravity(
        seed=seed,
        fold_index=fold_index,
        methods=_METHODS_ANTIGRAVITY,
        original={"C0": c0.normal_scores, "E12": ranks["normal"], "C12": c12_normal},
        primary={
            "C0": c0.primary_scores,
            "E12": {change: ranks[change] for change in _CHANGE_TYPES_ANTIGRAVITY},
            "C12": c12_primary,
        },
        fit_early_peak_lags=e12.fit_early_peak_lags,
        fit_late_peak_lags=e12.fit_late_peak_lags,
        fit_peak_presence=e12.fit_peak_presence,
        fit_shifts=e12.fit_shifts,
        fit_raw_shift=e12.fit_raw_shift,
        query_early_peak_lags=early,
        query_late_peak_lags=late,
        query_peak_presence=presence,
        query_shifts=shifts,
        query_raw_shift=raw,
        fit_normal_max_raw_shift=e12.fit_normal_max_raw_shift,
        confirmed=confirmed,
        input_train140=train,
        input_fit_indices=fit,
        input_query_indices=query,
        backend="numba",
    )


__all__ = (
    "FoldMethodScoresAntigravity",
    "_combine_c12_numba_antigravity",
    "score_fold_methods_antigravity",
)
