"""Immutable C0/S10/C10 fold scoring for Exp468 Task 4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .acceleration_antigravity import (
    AccelerationAssessmentAntigravity,
    BackendAssessmentAntigravity,
)
from .backend_selection_antigravity import (
    BackendDecisionAntigravity,
    finalize_backend_decision_antigravity,
)
from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .local_stft_antigravity import score_local_stft_antigravity
from .predecessor_antigravity import score_c0_fold_antigravity
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = ("C0", "S10", "C10")
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY
_APPROVAL_SHA256_ANTIGRAVITY = (
    "6780edb2683c992b3df361c72e15542a6e974cf3e7cbbc0df65df2739c0203a9"
)
_BENCHMARK_PATH_ANTIGRAVITY = "scratch/exp468_acceleration_benchmark_antigravity.json"
_APPROVAL_PATH_ANTIGRAVITY = (
    "docs/superpowers/specs/"
    "2026-08-24-exp468-numpy-fallback-antigravity-design-amendment.html"
)


def _approved_backend_decision_antigravity() -> BackendDecisionAntigravity:
    """Rehydrate and verify the checked Task 3 operational decision once."""

    root = Path(__file__).resolve().parents[2]
    payload = json.loads((root / _BENCHMARK_PATH_ANTIGRAVITY).read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise ValueError("Task 3 benchmark payload must be a JSON object")
    authorization = payload.get("fallback_authorization")
    if type(authorization) is not dict:
        raise ValueError("Task 3 fallback authorization is missing")
    if authorization.get("approval_document_path") != _APPROVAL_PATH_ANTIGRAVITY:
        raise ValueError("Task 3 approval document path drifted")
    recorded_sha = authorization.get("approval_document_sha256")
    actual_sha = hashlib.sha256((root / _APPROVAL_PATH_ANTIGRAVITY).read_bytes()).hexdigest()
    if recorded_sha != _APPROVAL_SHA256_ANTIGRAVITY or actual_sha != recorded_sha:
        raise ValueError("Task 3 fallback approval SHA-256 drifted")
    if (
        payload.get("status") != "TASK_3_COMPLETE_NUMPY_FALLBACK_AUTHORIZED"
        or payload.get("gate_passed") is not False
        or payload.get("fallback_authorized") is not True
        or payload.get("selected_backend") != "numpy"
        or authorization.get("authorized") is not True
    ):
        raise ValueError("Task 3 approved NumPy fallback decision drifted")

    backend_payloads = payload.get("backends")
    workload = payload.get("workload")
    timing = payload.get("timing_protocol")
    if type(backend_payloads) is not dict or type(workload) is not dict or type(timing) is not dict:
        raise ValueError("Task 3 assessment evidence is incomplete")
    backends = {
        name: BackendAssessmentAntigravity(
            **{
                field.name: backend_payload[field.name]
                for field in fields(BackendAssessmentAntigravity)
            },
        )
        for name, backend_payload in backend_payloads.items()
    }
    assessment = AccelerationAssessmentAntigravity(
        required_speedup=timing["required_speedup"],
        workload_counts={
            name: workload[name]
            for name in ("fit_raw", "query_raw", "transformed_rows", "frames")
        },
        backends=backends,
        selected_backend=payload["selected_backend"],
        gate_passed=payload["gate_passed"],
    )
    decision = finalize_backend_decision_antigravity(
        assessment,
        fallback_approval_sha256=recorded_sha,
    )
    if (
        decision.selected_backend != "numpy"
        or decision.acceleration_gate_passed is not False
        or decision.fallback_authorized is not True
        or decision.approval_evidence_sha256 != _APPROVAL_SHA256_ANTIGRAVITY
    ):
        raise ValueError("Task 3 decision cannot authorize Task 4 NumPy scoring")
    return decision


APPROVED_BACKEND_DECISION_ANTIGRAVITY = _approved_backend_decision_antigravity()


def _owned_train_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be an exact ndarray")
    if values.shape != (
        FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
        FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
    ):
        raise ValueError(f"{name} must have exact shape (69, 140)")
    if values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use exact float64 dtype")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
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
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if values.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must use exact int64 dtype")
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
        raise TypeError(f"{name} must be an exact ndarray")
    if values.shape != (length,):
        raise ValueError(f"{name} must have exact shape ({length},)")
    if values.dtype != dtype:
        raise ValueError(f"{name} has the wrong exact dtype")
    result = np.array(values, dtype=dtype, copy=True, order="C")
    if dtype == np.dtype(np.float64) and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
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
    """One fold's immutable methods, raw evidence, and source-free replay input."""

    seed: int
    fold_index: int
    methods: tuple[str, str, str]
    original: Mapping[str, np.ndarray]
    primary: Mapping[str, Mapping[str, np.ndarray]]
    fit_raw_js: np.ndarray
    query_raw_js: Mapping[str, np.ndarray]
    fit_normal_max_raw_js: float
    confirmed: Mapping[str, np.ndarray]
    input_train140: np.ndarray
    input_fit_indices: np.ndarray
    input_query_indices: np.ndarray
    backend: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG_ANTIGRAVITY.seeds:
            raise ValueError("seed must be one of the two frozen Exp468 seeds")
        if (
            type(self.fold_index) is not int
            or not 0 <= self.fold_index < FROZEN_CONFIG_ANTIGRAVITY.folds
        ):
            raise ValueError("fold_index must be one of the five frozen Exp468 folds")
        if type(self.methods) is not tuple or self.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("methods must preserve frozen C0/S10/C10 order")
        if self.backend != "numpy":
            raise ValueError("backend provenance must be the approved numpy decision")

        train = _owned_train_antigravity(self.input_train140, "input_train140")
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

        fit_raw = _owned_vector_antigravity(
            self.fit_raw_js,
            "fit_raw_js",
            length=len(fit),
            dtype=np.dtype(np.float64),
        )
        maximum = self.fit_normal_max_raw_js
        if type(maximum) is not float or not isfinite(maximum):
            raise ValueError("fit_normal_max_raw_js must be an exact finite float")
        if maximum != float(np.max(fit_raw)):
            raise ValueError("fit_normal_max_raw_js must equal max(fit_raw_js)")
        query_raw_source = _ordered_mapping_antigravity(
            self.query_raw_js,
            _SCOPES_ANTIGRAVITY,
            "query_raw_js",
        )
        query_raw = {
            scope: _owned_vector_antigravity(
                query_raw_source[scope],
                f"query_raw_js[{scope}]",
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

        for scope in _SCOPES_ANTIGRAVITY:
            expected_confirmation = (query_raw[scope] > 0.0) & (query_raw[scope] > maximum)
            if not np.array_equal(confirmed[scope], expected_confirmation):
                raise ValueError("confirmation must follow the strict raw-JS boundary")
            expected_rank = (
                1.0
                + np.count_nonzero(
                    fit_raw[:, None] <= query_raw[scope][None, :],
                    axis=0,
                )
            ) / float(len(fit_raw) + 1)
            s10 = original["S10"] if scope == "normal" else primary["S10"][scope]
            if not np.array_equal(s10, np.asarray(expected_rank, dtype=np.float64)):
                raise ValueError("S10 rank must be derived exactly from retained raw JS")
            c0 = original["C0"] if scope == "normal" else primary["C0"][scope]
            c10 = original["C10"] if scope == "normal" else primary["C10"][scope]
            expected_c10 = np.where(confirmed[scope], np.maximum(c0, s10), c0)
            if not np.array_equal(c10, expected_c10):
                raise ValueError("C10 must be confirmed max(C0, S10) with exact C0 fallback")
            if not np.array_equal(c10[~confirmed[scope]], c0[~confirmed[scope]]):
                raise ValueError("unconfirmed C10 must be array-equal to C0")

        object.__setattr__(self, "original", MappingProxyType(original))
        object.__setattr__(self, "primary", MappingProxyType(primary))
        object.__setattr__(self, "fit_raw_js", fit_raw)
        object.__setattr__(self, "query_raw_js", MappingProxyType(query_raw))
        object.__setattr__(self, "fit_normal_max_raw_js", maximum)
        object.__setattr__(self, "confirmed", MappingProxyType(confirmed))
        object.__setattr__(self, "input_train140", train)
        object.__setattr__(self, "input_fit_indices", fit)
        object.__setattr__(self, "input_query_indices", query)


def _scope_rows_antigravity(train: np.ndarray, query: np.ndarray) -> Mapping[str, np.ndarray]:
    generated = tuple(primary_rows_antigravity(row) for row in train[query])
    return MappingProxyType(
        {
            "normal": train[query],
            **{
                change_type: np.stack(
                    tuple(changes[change_type] for changes in generated),
                )
                for change_type in _CHANGE_TYPES_ANTIGRAVITY
            },
        },
    )


def score_fold_methods_antigravity(
    train140: np.ndarray,
    fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    authority: object,
) -> FoldMethodScoresAntigravity:
    """Score one fold with fixed C0 and the approved NumPy-only S10 path."""

    decision = APPROVED_BACKEND_DECISION_ANTIGRAVITY
    if (
        decision.selected_backend != "numpy"
        or decision.acceleration_gate_passed is not False
        or decision.fallback_authorized is not True
        or decision.approval_evidence_sha256 != _APPROVAL_SHA256_ANTIGRAVITY
    ):
        raise PermissionError("Task 4 requires the approved NumPy fallback decision")
    train = _owned_train_antigravity(train140, "train140")
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
    scope_rows = _scope_rows_antigravity(train, query)
    combined_query = np.concatenate(
        tuple(scope_rows[scope] for scope in _SCOPES_ANTIGRAVITY),
        axis=0,
    )
    spectral = score_local_stft_antigravity(
        train[fit],
        combined_query,
        backend="numpy",
    )
    if spectral.backend != "numpy":
        raise ValueError("Task 4 S10 scorer returned non-NumPy provenance")
    width = len(query)
    query_raw = {
        scope: spectral.query_raw_js[index * width : (index + 1) * width]
        for index, scope in enumerate(_SCOPES_ANTIGRAVITY)
    }
    ranks = {
        scope: spectral.query_ranks[index * width : (index + 1) * width]
        for index, scope in enumerate(_SCOPES_ANTIGRAVITY)
    }
    confirmed = {
        scope: spectral.confirmation_mask[index * width : (index + 1) * width]
        for index, scope in enumerate(_SCOPES_ANTIGRAVITY)
    }

    c10_normal = np.where(
        confirmed["normal"],
        np.maximum(c0.normal_scores, ranks["normal"]),
        c0.normal_scores,
    )
    c10_primary = {
        change_type: np.where(
            confirmed[change_type],
            np.maximum(c0.primary_scores[change_type], ranks[change_type]),
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
                "S10": ranks["normal"],
                "C10": c10_normal,
            },
        ),
        primary=MappingProxyType(
            {
                "C0": c0.primary_scores,
                "S10": MappingProxyType(
                    {change_type: ranks[change_type] for change_type in _CHANGE_TYPES_ANTIGRAVITY},
                ),
                "C10": MappingProxyType(c10_primary),
            },
        ),
        fit_raw_js=spectral.fit_raw_js,
        query_raw_js=MappingProxyType(query_raw),
        fit_normal_max_raw_js=spectral.fit_normal_max_raw_js,
        confirmed=MappingProxyType(confirmed),
        input_train140=train,
        input_fit_indices=fit,
        input_query_indices=query,
        backend="numpy",
    )


__all__ = (
    "APPROVED_BACKEND_DECISION_ANTIGRAVITY",
    "FoldMethodScoresAntigravity",
    "score_fold_methods_antigravity",
)
