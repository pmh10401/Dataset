"""Pure synthetic-record qualification and Numba pair-repair accounting."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numba
import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp471FoldResultAntigravity,
    Exp471QualificationAntigravity,
    PairRepairCountsAntigravity,
)
from .scoring_antigravity import FoldMethodScoresAntigravity


_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _owned_vector_antigravity(values: object, name: str) -> np.ndarray:
    if (
        type(values) is not np.ndarray
        or values.ndim != 1
        or values.size == 0
        or values.dtype != np.dtype(np.float64)
    ):
        raise ValueError(f"{name} must be a nonempty exact float64 vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def validate_four_score_vectors_antigravity(
    c0_normal: object,
    c0_changed: object,
    c12_normal: object,
    c12_changed: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    before_normal = _owned_vector_antigravity(c0_normal, "c0_normal")
    before_changed = _owned_vector_antigravity(c0_changed, "c0_changed")
    after_normal = _owned_vector_antigravity(c12_normal, "c12_normal")
    after_changed = _owned_vector_antigravity(c12_changed, "c12_changed")
    if before_normal.size != after_normal.size or before_changed.size != after_changed.size:
        raise ValueError("C0/C12 normal and changed lengths must match")
    return before_normal, before_changed, after_normal, after_changed


@numba.njit(cache=False, fastmath=False)
def _count_pair_repairs_numba_antigravity(
    c0_normal: np.ndarray,
    c0_changed: np.ndarray,
    c12_normal: np.ndarray,
    c12_changed: np.ndarray,
) -> tuple[int, int]:
    fixed = 0
    broken = 0
    for normal_index in range(c0_normal.size):
        for changed_index in range(c0_changed.size):
            before = (
                1.0
                if c0_changed[changed_index] > c0_normal[normal_index]
                else (0.5 if c0_changed[changed_index] == c0_normal[normal_index] else 0.0)
            )
            after = (
                1.0
                if c12_changed[changed_index] > c12_normal[normal_index]
                else (0.5 if c12_changed[changed_index] == c12_normal[normal_index] else 0.0)
            )
            if after > before:
                fixed += 1
            elif after < before:
                broken += 1
    return fixed, broken


def count_pair_repairs_antigravity(
    c0_normal: object,
    c0_changed: object,
    c12_normal: object,
    c12_changed: object,
) -> PairRepairCountsAntigravity:
    values = validate_four_score_vectors_antigravity(
        c0_normal, c0_changed, c12_normal, c12_changed
    )
    fixed, broken = _count_pair_repairs_numba_antigravity(*values)
    return PairRepairCountsAntigravity(fixed=int(fixed), broken=int(broken))


@numba.njit(cache=False, fastmath=False)
def _count_confirmed_numba_antigravity(confirmed: np.ndarray) -> int:
    count = 0
    for index in range(confirmed.size):
        if confirmed[index]:
            count += 1
    return count


def _fallback_bytes_exact_antigravity(record: FoldMethodScoresAntigravity, scope: str) -> bool:
    c0 = record.original["C0"] if scope == "normal" else record.primary["C0"][scope]
    rank = record.original["E12"] if scope == "normal" else record.primary["E12"][scope]
    c12 = record.original["C12"] if scope == "normal" else record.primary["C12"][scope]
    fallback = np.logical_not(record.confirmed[scope]) | (rank <= c0)
    return c12[fallback].tobytes(order="C") == c0[fallback].tobytes(order="C")


def blocked_fold_indices_antigravity() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for query_source in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds):
        query = np.array(query_source, dtype=np.int64, copy=True, order="C")
        fit = np.array(
            np.setdiff1d(all_indices, query, assume_unique=True),
            dtype=np.int64,
            copy=True,
            order="C",
        )
        fit.setflags(write=False)
        query.setflags(write=False)
        result.append((fit, query))
    return tuple(result)


def _normalise_records_antigravity(records: object) -> tuple[FoldMethodScoresAntigravity, ...]:
    expected = tuple(
        (seed, fold)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if not isinstance(records, Mapping) or tuple(records) != expected:
        raise ValueError("records must preserve exact seed-major/fold-major order")
    values = tuple(records.values())
    if any(type(record) is not FoldMethodScoresAntigravity for record in values):
        raise ValueError("records must contain exact FoldMethodScoresAntigravity values")
    folds = blocked_fold_indices_antigravity()
    train_bytes = values[0].input_train140.tobytes(order="C")
    for identity, record in zip(expected, values, strict=True):
        if identity != (record.seed, record.fold_index):
            raise ValueError("record identity drifted")
        if record.methods != ("C0", "E12", "C12") or record.backend != "numba":
            raise ValueError("record method or Numba provenance drifted")
        fit, query = folds[record.fold_index]
        if not np.array_equal(record.input_fit_indices, fit) or not np.array_equal(
            record.input_query_indices, query
        ):
            raise ValueError("record blocked fold arrays drifted")
        if record.input_train140.tobytes(order="C") != train_bytes:
            raise ValueError("record TRAIN bytes drifted")
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        out_of_fold = np.concatenate(
            tuple(record.input_query_indices for record in values if record.seed == seed)
        )
        if not np.array_equal(out_of_fold, np.arange(69, dtype=np.int64)):
            raise ValueError("seed OOF blocks must cover 0..68 exactly once")
    return values


def qualify_records_antigravity(records: object) -> Exp471QualificationAntigravity:
    """Aggregate exactly ten source records without loading or rescoring data."""
    values = _normalise_records_antigravity(records)
    fold_results: list[Exp471FoldResultAntigravity] = []
    period_fixed = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    period_broken = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    protected_broken = {change: 0 for change in _CHANGE_TYPES_ANTIGRAVITY[:3]}
    normal_confirmations = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    fallback_exact = True
    finite = True

    for record in values:
        c0_auc = {
            change: float(
                mann_whitney_auc_antigravity(
                    record.original["C0"], record.primary["C0"][change]
                )
            )
            for change in _CHANGE_TYPES_ANTIGRAVITY
        }
        c12_auc = {
            change: float(
                mann_whitney_auc_antigravity(
                    record.original["C12"], record.primary["C12"][change]
                )
            )
            for change in _CHANGE_TYPES_ANTIGRAVITY
        }
        repairs = {
            change: count_pair_repairs_antigravity(
                record.original["C0"],
                record.primary["C0"][change],
                record.original["C12"],
                record.primary["C12"][change],
            )
            for change in _CHANGE_TYPES_ANTIGRAVITY
        }
        normal_count = int(_count_confirmed_numba_antigravity(record.confirmed["normal"]))
        score_arrays = (
            record.fit_raw_shift,
            *record.query_raw_shift.values(),
            *record.original.values(),
            *(array for method in record.primary.values() for array in method.values()),
        )
        record_finite = bool(all(np.isfinite(array).all() for array in score_arrays))
        record_fallback = all(
            _fallback_bytes_exact_antigravity(record, scope) for scope in _SCOPES_ANTIGRAVITY
        )
        reasons: tuple[str, ...] = ()
        if not record_finite:
            reasons += ("nonfinite_scores",)
        if not record_fallback:
            reasons += ("fallback_not_exact",)
        fold_results.append(
            Exp471FoldResultAntigravity(
                seed=record.seed,
                fold_index=record.fold_index,
                c0_per_type_auc=c0_auc,
                c12_per_type_auc=c12_auc,
                pair_repairs=repairs,
                normal_confirmations=normal_count,
                fallback_exact=record_fallback,
                finite=record_finite,
                passed=not reasons,
                failure_reasons=reasons,
            )
        )
        period_fixed[record.seed] += repairs["period_change"].fixed
        period_broken[record.seed] += repairs["period_change"].broken
        for change in protected_broken:
            protected_broken[change] += repairs[change].broken
        normal_confirmations[record.seed] += normal_count
        fallback_exact = fallback_exact and record_fallback
        finite = finite and record_finite

    return Exp471QualificationAntigravity(
        fold_results=tuple(fold_results),
        period_fixed=MappingProxyType(period_fixed),
        period_broken=MappingProxyType(period_broken),
        protected_broken=MappingProxyType(protected_broken),
        normal_confirmations=MappingProxyType(normal_confirmations),
        fallback_exact=fallback_exact,
        finite=finite,
        label_query_count=0,
        artifact_map_sha256=None,
    )


__all__ = (
    "_count_confirmed_numba_antigravity",
    "_count_pair_repairs_numba_antigravity",
    "blocked_fold_indices_antigravity",
    "count_pair_repairs_antigravity",
    "qualify_records_antigravity",
    "validate_four_score_vectors_antigravity",
)
