"""Assemble immutable Exp466 C0/P9/C9 fold scores and replay evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

import numpy as np

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .phase_registration_antigravity import score_phase_registration_antigravity
from .predecessor_antigravity import (
    Exp465AuthorityAntigravity,
    score_c0_fold_antigravity,
)
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = ("C0", "P9", "C9")
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _owned_matrix_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 140:
        raise ValueError(f"{name} must have at least three rows and width 140")
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _owned_indices_antigravity(
    values: object,
    name: str,
    *,
    minimum_size: int,
    row_count: int,
) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if values.ndim != 1 or values.size < minimum_size:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if values.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must use int64")
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must contain unique indices")
    if int(np.min(result)) < 0 or int(np.max(result)) >= row_count:
        raise ValueError(f"{name} contains an out-of-range index")
    result.setflags(write=False)
    return result


def _owned_vector_antigravity(
    values: object,
    name: str,
    *,
    length: int,
    dtype: np.dtype,
) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise ValueError(f"{name} must be an exact ndarray")
    if values.ndim != 1 or values.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if values.dtype != dtype:
        raise ValueError(f"{name} has the wrong dtype")
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if dtype == np.dtype(np.float64) and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    result.setflags(write=False)
    return result


def _ordered_mapping_antigravity(
    values: object,
    expected: tuple[str, ...],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if tuple(values) != expected:
        raise ValueError(f"{name} must preserve exact order")
    return values


@dataclass(frozen=True, slots=True)
class FoldMethodScoresAntigravity:
    """One frozen fold with scores plus source-free replay inputs."""

    seed: int
    fold_index: int
    methods: tuple[str, str, str]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    fit_margins: np.ndarray
    query_margins: Mapping[str, np.ndarray]
    fit_normal_max_margin: float
    confirmed: Mapping[str, np.ndarray]
    input_train140: np.ndarray
    input_fit_indices: np.ndarray
    input_query_indices: np.ndarray

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed must be one of the frozen Exp466 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index must be one of the frozen Exp466 folds")
        if type(self.methods) is not tuple or self.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("methods must preserve frozen C0/P9/C9 order")

        train = _owned_matrix_antigravity(self.input_train140, "input_train140")
        fit = _owned_indices_antigravity(
            self.input_fit_indices,
            "input_fit_indices",
            minimum_size=2,
            row_count=len(train),
        )
        query = _owned_indices_antigravity(
            self.input_query_indices,
            "input_query_indices",
            minimum_size=1,
            row_count=len(train),
        )
        if set(int(value) for value in fit) & set(int(value) for value in query):
            raise ValueError("input fit/query indices must be disjoint")

        original_source = _ordered_mapping_antigravity(
            self.original,
            _METHODS_ANTIGRAVITY,
            "original",
        )
        original = {
            method: _owned_vector_antigravity(
                original_source[method],
                f"original[{method}]",
                length=len(query),
                dtype=np.dtype(np.float64),
            )
            for method in _METHODS_ANTIGRAVITY
        }
        primary_source = _ordered_mapping_antigravity(
            self.primary,
            _METHODS_ANTIGRAVITY,
            "primary",
        )
        primary: dict[str, Mapping[str, np.ndarray]] = {}
        for method in _METHODS_ANTIGRAVITY:
            method_source = _ordered_mapping_antigravity(
                primary_source[method],
                _CHANGE_TYPES_ANTIGRAVITY,
                f"primary[{method}]",
            )
            primary[method] = MappingProxyType(
                {
                    change_type: _owned_vector_antigravity(
                        method_source[change_type],
                        f"primary[{method}][{change_type}]",
                        length=len(query),
                        dtype=np.dtype(np.float64),
                    )
                    for change_type in _CHANGE_TYPES_ANTIGRAVITY
                },
            )

        fit_margins = _owned_vector_antigravity(
            self.fit_margins,
            "fit_margins",
            length=len(fit),
            dtype=np.dtype(np.float64),
        )
        threshold = self.fit_normal_max_margin
        if type(threshold) is not float or not isfinite(threshold):
            raise ValueError("fit_normal_max_margin must be a finite float")
        if threshold != float(np.max(fit_margins)):
            raise ValueError("fit_normal_max_margin must equal max(fit_margins)")
        margins_source = _ordered_mapping_antigravity(
            self.query_margins,
            _SCOPES_ANTIGRAVITY,
            "query_margins",
        )
        margins = {
            scope: _owned_vector_antigravity(
                margins_source[scope],
                f"query_margins[{scope}]",
                length=len(query),
                dtype=np.dtype(np.float64),
            )
            for scope in _SCOPES_ANTIGRAVITY
        }
        confirmed_source = _ordered_mapping_antigravity(
            self.confirmed,
            _SCOPES_ANTIGRAVITY,
            "confirmed",
        )
        confirmed = {
            scope: _owned_vector_antigravity(
                confirmed_source[scope],
                f"confirmed[{scope}]",
                length=len(query),
                dtype=np.dtype(np.bool_),
            )
            for scope in _SCOPES_ANTIGRAVITY
        }

        object.__setattr__(self, "original", MappingProxyType(original))
        object.__setattr__(self, "primary", MappingProxyType(primary))
        object.__setattr__(self, "fit_margins", fit_margins)
        object.__setattr__(self, "query_margins", MappingProxyType(margins))
        object.__setattr__(self, "confirmed", MappingProxyType(confirmed))
        object.__setattr__(self, "input_train140", train)
        object.__setattr__(self, "input_fit_indices", fit)
        object.__setattr__(self, "input_query_indices", query)


def score_fold_methods_antigravity(
    train140: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: Exp465AuthorityAntigravity,
) -> FoldMethodScoresAntigravity:
    """Score the original query and four frozen changes with C0/P9/C9."""

    train = _owned_matrix_antigravity(train140, "train140")
    fit = _owned_indices_antigravity(
        fit_indices,
        "fit_indices",
        minimum_size=2,
        row_count=len(train),
    )
    query = _owned_indices_antigravity(
        query_indices,
        "query_indices",
        minimum_size=1,
        row_count=len(train),
    )
    if set(int(value) for value in fit) & set(int(value) for value in query):
        raise ValueError("fit/query indices must be disjoint")

    c0 = score_c0_fold_antigravity(
        train,
        fit,
        query,
        seed=seed,
        fold_index=fold_index,
        authority=authority,
    )
    generated = tuple(primary_rows_antigravity(row) for row in train[query])
    query_rows = {
        "normal": train[query],
        **{
            change_type: np.stack(
                tuple(changes[change_type] for changes in generated),
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
    }
    phases = {
        scope: score_phase_registration_antigravity(train[fit], query_rows[scope])
        for scope in _SCOPES_ANTIGRAVITY
    }
    normal_phase = phases["normal"]
    for scope in _CHANGE_TYPES_ANTIGRAVITY:
        if not np.array_equal(phases[scope].fit_margins, normal_phase.fit_margins):
            raise ValueError("phase scopes produced different fit calibration")

    p9_normal = normal_phase.ranks
    c9_normal = np.where(
        normal_phase.confirmed,
        np.maximum(c0.normal_scores, p9_normal),
        c0.normal_scores,
    )
    p9_primary = {
        change_type: phases[change_type].ranks
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    c9_primary = {
        change_type: np.where(
            phases[change_type].confirmed,
            np.maximum(c0.primary_scores[change_type], p9_primary[change_type]),
            c0.primary_scores[change_type],
        )
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    }
    return FoldMethodScoresAntigravity(
        seed=seed,
        fold_index=fold_index,
        methods=_METHODS_ANTIGRAVITY,
        original=MappingProxyType(
            {
                "C0": c0.normal_scores,
                "P9": p9_normal,
                "C9": c9_normal,
            },
        ),
        primary=MappingProxyType(
            {
                "C0": c0.primary_scores,
                "P9": MappingProxyType(p9_primary),
                "C9": MappingProxyType(c9_primary),
            },
        ),
        fit_margins=normal_phase.fit_margins,
        query_margins=MappingProxyType(
            {scope: phases[scope].query_margins for scope in _SCOPES_ANTIGRAVITY},
        ),
        fit_normal_max_margin=normal_phase.fit_normal_max_margin,
        confirmed=MappingProxyType(
            {scope: phases[scope].confirmed for scope in _SCOPES_ANTIGRAVITY},
        ),
        input_train140=train,
        input_fit_indices=fit,
        input_query_indices=query,
    )


__all__ = ("FoldMethodScoresAntigravity", "score_fold_methods_antigravity")
