"""Pure synthetic-record qualification and compiled pair-repair accounting."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numba
import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import mann_whitney_auc_antigravity

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import Exp470FoldResultAntigravity, Exp470QualificationAntigravity, PairRepairCountsAntigravity
from .scoring_antigravity import FoldMethodScoresAntigravity


_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types


def _owned_vector_antigravity(values: object, name: str) -> np.ndarray:
    if type(values) is not np.ndarray or values.ndim != 1 or values.size == 0 or values.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must be a nonempty exact float64 vector")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def validate_four_score_vectors_antigravity(c0_normal: object, c0_changed: object, c11_normal: object, c11_changed: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return frozen validated vectors in the contract's argument order."""
    before_normal = _owned_vector_antigravity(c0_normal, "c0_normal")
    before_changed = _owned_vector_antigravity(c0_changed, "c0_changed")
    after_normal = _owned_vector_antigravity(c11_normal, "c11_normal")
    after_changed = _owned_vector_antigravity(c11_changed, "c11_changed")
    if before_normal.size != after_normal.size or before_changed.size != after_changed.size:
        raise ValueError("C0/C11 normal and changed lengths must match")
    return before_normal, before_changed, after_normal, after_changed


@numba.njit(cache=False, fastmath=False)
def _count_pair_repairs_numba_antigravity(before_normal: np.ndarray, before_changed: np.ndarray, after_normal: np.ndarray, after_changed: np.ndarray) -> tuple[int, int]:
    fixed = 0
    broken = 0
    for normal_index in range(before_normal.size):
        for changed_index in range(before_changed.size):
            before_credit = 1.0 if before_changed[changed_index] > before_normal[normal_index] else (0.5 if before_changed[changed_index] == before_normal[normal_index] else 0.0)
            after_credit = 1.0 if after_changed[changed_index] > after_normal[normal_index] else (0.5 if after_changed[changed_index] == after_normal[normal_index] else 0.0)
            if after_credit > before_credit:
                fixed += 1
            elif after_credit < before_credit:
                broken += 1
    return fixed, broken


def count_pair_repairs_antigravity(c0_normal: object, c0_changed: object, c11_normal: object, c11_changed: object) -> PairRepairCountsAntigravity:
    values = validate_four_score_vectors_antigravity(c0_normal, c0_changed, c11_normal, c11_changed)
    fixed, broken = _count_pair_repairs_numba_antigravity(*values)
    return PairRepairCountsAntigravity(fixed=int(fixed), broken=int(broken))


def blocked_fold_indices_antigravity() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for query_source in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds):
        query = np.array(query_source, dtype=np.int64, copy=True, order="C")
        fit = np.array(np.setdiff1d(all_indices, query, assume_unique=True), dtype=np.int64, copy=True, order="C")
        fit.setflags(write=False)
        query.setflags(write=False)
        result.append((fit, query))
    return tuple(result)


def _normalise_records_antigravity(records: object) -> tuple[FoldMethodScoresAntigravity, ...]:
    expected = tuple((seed, fold) for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds))
    if not isinstance(records, Mapping) or tuple(records) != expected:
        raise ValueError("records must preserve exact seed-major/fold-major order")
    values = tuple(records.values())
    if any(type(record) is not FoldMethodScoresAntigravity for record in values):
        raise ValueError("records must contain exact FoldMethodScoresAntigravity values")
    for identity, record in zip(expected, values, strict=True):
        if identity != (record.seed, record.fold_index) or record.methods != ("C0", "E11", "C11") or record.backend != "numba":
            raise ValueError("record identity or Numba provenance drifted")
    train_bytes = values[0].input_train140.tobytes(order="C")
    folds = blocked_fold_indices_antigravity()
    for record in values:
        fit, query = folds[record.fold_index]
        if not np.array_equal(record.input_fit_indices, fit) or not np.array_equal(record.input_query_indices, query):
            raise ValueError("record blocked fold arrays drifted")
        if record.input_train140.tobytes(order="C") != train_bytes:
            raise ValueError("record TRAIN bytes drifted")
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        oof = np.concatenate(tuple(record.input_query_indices for record in values if record.seed == seed))
        if not np.array_equal(oof, np.arange(69, dtype=np.int64)):
            raise ValueError("seed OOF blocks must cover 0..68 exactly once")
    return values


def qualify_records_antigravity(records: object) -> Exp470QualificationAntigravity:
    """Aggregate exactly ten source records; no loader, rescore, or label access occurs."""
    values = _normalise_records_antigravity(records)
    fold_results: list[Exp470FoldResultAntigravity] = []
    period_fixed = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    period_broken = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    protected = {name: 0 for name in _CHANGE_TYPES_ANTIGRAVITY[:3]}
    normal_confirmations = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    fallback_exact = True
    finite = True
    for record in values:
        c0_auc = {name: float(mann_whitney_auc_antigravity(record.original["C0"], record.primary["C0"][name])) for name in _CHANGE_TYPES_ANTIGRAVITY}
        c11_auc = {name: float(mann_whitney_auc_antigravity(record.original["C11"], record.primary["C11"][name])) for name in _CHANGE_TYPES_ANTIGRAVITY}
        repairs = {name: count_pair_repairs_antigravity(record.original["C0"], record.primary["C0"][name], record.original["C11"], record.primary["C11"][name]) for name in _CHANGE_TYPES_ANTIGRAVITY}
        normal_count = int(np.count_nonzero(record.confirmed["normal"]))
        record_finite = bool(all(np.isfinite(vector).all() for vector in (record.fit_raw_entropy, *record.query_raw_entropy.values(), *record.original.values(), *(vector for method in record.primary.values() for vector in method.values()))))
        record_fallback = all(np.array_equal((record.original["C11"] if scope == "normal" else record.primary["C11"][scope])[~record.confirmed[scope]], (record.original["C0"] if scope == "normal" else record.primary["C0"][scope])[~record.confirmed[scope]]) for scope in ("normal",) + _CHANGE_TYPES_ANTIGRAVITY)
        reasons = () if record_finite and record_fallback else (("nonfinite_scores" if not record_finite else "fallback_not_exact"),)
        fold_results.append(Exp470FoldResultAntigravity(seed=record.seed, fold_index=record.fold_index, c0_per_type_auc=c0_auc, c11_per_type_auc=c11_auc, pair_repairs=repairs, normal_confirmations=normal_count, fallback_exact=record_fallback, finite=record_finite, passed=not reasons, failure_reasons=reasons))
        period_fixed[record.seed] += repairs["period_change"].fixed
        period_broken[record.seed] += repairs["period_change"].broken
        for name in protected:
            protected[name] += repairs[name].broken
        normal_confirmations[record.seed] += normal_count
        fallback_exact = fallback_exact and record_fallback
        finite = finite and record_finite
    return Exp470QualificationAntigravity(fold_results=tuple(fold_results), period_fixed=MappingProxyType(period_fixed), period_broken=MappingProxyType(period_broken), protected_broken=MappingProxyType(protected), normal_confirmations=MappingProxyType(normal_confirmations), fallback_exact=fallback_exact, finite=finite, label_query_count=0, artifact_map_sha256=None)


__all__ = ("_count_pair_repairs_numba_antigravity", "blocked_fold_indices_antigravity", "count_pair_repairs_antigravity", "qualify_records_antigravity", "validate_four_score_vectors_antigravity")
