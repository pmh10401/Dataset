"""Exact ten-record qualification for Exp466 C9."""

from __future__ import annotations

from collections.abc import Mapping
from math import fsum
from types import MappingProxyType

import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp466FoldResultAntigravity,
    Exp466QualificationAntigravity,
)
from .phase_registration_antigravity import score_phase_registration_antigravity
from .predecessor_antigravity import (
    capture_exp465_authority_antigravity,
    score_c0_fold_antigravity,
)
from .scoring_antigravity import FoldMethodScoresAntigravity
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = ("C0", "P9", "C9")
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _failure_reasons_antigravity(
    per_type_auc: tuple[float, float, float, float],
    mean_auc: float,
    all_scores_finite: bool,
    c0_gradual_auc: float,
    c0_period_auc: float,
    c9_gradual_auc: float,
    c9_period_auc: float,
    period_confirmations: int,
    normal_confirmations: int,
) -> tuple[str, ...]:
    """Return every failed frozen C9 gate in deterministic order."""

    reasons: list[str] = []
    if not all_scores_finite:
        reasons.append("nonfinite_scores")
    for change_type, auc in zip(_CHANGE_TYPES_ANTIGRAVITY, per_type_auc, strict=True):
        if auc < FROZEN_CONFIG_ANTIGRAVITY.per_type_auc_min:
            reasons.append(f"{change_type}_auc_below_0.85")
    if mean_auc < FROZEN_CONFIG_ANTIGRAVITY.mean_auc_min:
        reasons.append("mean_auc_below_0.90")
    if c9_period_auc <= c0_period_auc:
        reasons.append("period_auc_not_strictly_above_c0")
    if c9_gradual_auc < c0_gradual_auc:
        reasons.append("gradual_auc_below_c0")
    if period_confirmations == 0:
        reasons.append("period_confirmation_missing")
    if normal_confirmations != 0:
        reasons.append("normal_confirmation_present")
    return tuple(reasons)


def _all_score_arrays_antigravity(
    record: FoldMethodScoresAntigravity,
) -> tuple[np.ndarray, ...]:
    return (
        record.fit_margins,
        *record.query_margins.values(),
        *record.original.values(),
        *(values for method in record.primary.values() for values in method.values()),
    )


def _normalise_records_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
) -> tuple[tuple[tuple[int, int], FoldMethodScoresAntigravity], ...]:
    if not isinstance(records, Mapping):
        raise ValueError("records must be a mapping")
    expected = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if tuple(records) != expected:
        raise ValueError("records must preserve exact frozen seed-major order")
    items = tuple(records.items())
    for key, record in items:
        if type(record) is not FoldMethodScoresAntigravity:
            raise ValueError("records must contain exact FoldMethodScoresAntigravity values")
        if key != (record.seed, record.fold_index):
            raise ValueError("record key differs from its seed/fold identity")
        if record.methods != _METHODS_ANTIGRAVITY:
            raise ValueError("record methods differ from frozen C0/P9/C9 order")
        if any(not np.isfinite(values).all() for values in _all_score_arrays_antigravity(record)):
            raise ValueError("nonfinite_scores")

    expected_trials = np.arange(
        FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed,
        dtype=np.int64,
    )
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        oof = np.concatenate(
            tuple(
                record.input_query_indices
                for (identity, record) in items
                if identity[0] == seed
            ),
        )
        if len(oof) != len(expected_trials) or len(np.unique(oof)) != len(expected_trials):
            raise ValueError("each seed must cover exactly 69 unique OOF normal trials")
        if not np.array_equal(np.sort(oof), expected_trials):
            raise ValueError("each seed must cover exactly 69 unique OOF normal trials")

    first_train = items[0][1].input_train140
    expected_index_set = set(int(value) for value in expected_trials)
    for _key, record in items:
        if record.input_train140.shape != (69, 140):
            raise ValueError("qualification input train must be exactly 69x140")
        if record.input_train140.tobytes() != first_train.tobytes():
            raise ValueError("all records must retain the same train140 input")
        fit_set = set(int(value) for value in record.input_fit_indices)
        query_set = set(int(value) for value in record.input_query_indices)
        if fit_set | query_set != expected_index_set or fit_set & query_set:
            raise ValueError("each fold fit/query split must partition all 69 trials")
    return items


def _scope_rows_antigravity(
    record: FoldMethodScoresAntigravity,
) -> Mapping[str, np.ndarray]:
    train = record.input_train140
    query = record.input_query_indices
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


def _replay_record_antigravity(
    record: FoldMethodScoresAntigravity,
) -> None:
    authority = capture_exp465_authority_antigravity()
    c0 = score_c0_fold_antigravity(
        record.input_train140,
        record.input_fit_indices,
        record.input_query_indices,
        seed=record.seed,
        fold_index=record.fold_index,
        authority=authority,
    )
    if record.original["C0"].tobytes() != c0.normal_scores.tobytes():
        raise ValueError("C0 input replay differs from stored normal scores")
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        if (
            record.primary["C0"][change_type].tobytes()
            != c0.primary_scores[change_type].tobytes()
        ):
            raise ValueError("C0 input replay differs from stored primary scores")

    fit_rows = record.input_train140[record.input_fit_indices]
    for scope, query_rows in _scope_rows_antigravity(record).items():
        phase = score_phase_registration_antigravity(fit_rows, query_rows)
        p9 = record.original["P9"] if scope == "normal" else record.primary["P9"][scope]
        c0_scores = (
            record.original["C0"]
            if scope == "normal"
            else record.primary["C0"][scope]
        )
        c9 = record.original["C9"] if scope == "normal" else record.primary["C9"][scope]
        if (
            not np.array_equal(record.fit_margins, phase.fit_margins)
            or record.fit_normal_max_margin != phase.fit_normal_max_margin
            or not np.array_equal(record.query_margins[scope], phase.query_margins)
            or not np.array_equal(p9, phase.ranks)
            or not np.array_equal(record.confirmed[scope], phase.confirmed)
        ):
            raise ValueError("P9 raw evidence differs from retained input replay")
        expected_c9 = np.where(
            phase.confirmed,
            np.maximum(c0_scores, phase.ranks),
            c0_scores,
        )
        if not np.array_equal(c9, expected_c9):
            raise ValueError("C9 score differs from raw P9/C0 replay")


def _method_auc_antigravity(
    record: FoldMethodScoresAntigravity,
    method: str,
) -> tuple[tuple[float, float, float, float], float, bool]:
    arrays = (
        record.original[method],
        *(record.primary[method][change_type] for change_type in _CHANGE_TYPES_ANTIGRAVITY),
    )
    finite = bool(all(np.isfinite(values).all() for values in arrays))
    if not finite:
        return (0.0, 0.0, 0.0, 0.0), 0.0, False
    auc = tuple(
        float(
            mann_whitney_auc_antigravity(
                record.original[method],
                record.primary[method][change_type],
            ),
        )
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    )
    return auc, float(fsum(auc) / len(auc)), True


def qualify_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
) -> Exp466QualificationAntigravity:
    """Replay exactly ten fold records and emit frozen scalar C9 rows."""

    items = _normalise_records_antigravity(records)
    fold_results: list[Exp466FoldResultAntigravity] = []
    normal_confirmations = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    for (_seed_fold, record) in items:
        _replay_record_antigravity(record)
        c0_auc, _c0_mean, _c0_finite = _method_auc_antigravity(record, "C0")
        c9_auc, c9_mean, c9_finite = _method_auc_antigravity(record, "C9")
        period_confirmations = int(np.count_nonzero(record.confirmed["period_change"]))
        fold_normal_confirmations = int(np.count_nonzero(record.confirmed["normal"]))
        normal_confirmations[record.seed] += fold_normal_confirmations
        reasons = _failure_reasons_antigravity(
            per_type_auc=c9_auc,
            mean_auc=c9_mean,
            all_scores_finite=c9_finite,
            c0_gradual_auc=c0_auc[1],
            c0_period_auc=c0_auc[3],
            c9_gradual_auc=c9_auc[1],
            c9_period_auc=c9_auc[3],
            period_confirmations=period_confirmations,
            normal_confirmations=fold_normal_confirmations,
        )
        fold_results.append(
            Exp466FoldResultAntigravity(
                seed=record.seed,
                fold_index=record.fold_index,
                per_type_auc=c9_auc,
                mean_auc=c9_mean,
                c0_gradual_auc=c0_auc[1],
                c0_period_auc=c0_auc[3],
                c9_gradual_auc=c9_auc[1],
                c9_period_auc=c9_auc[3],
                period_confirmations=period_confirmations,
                all_scores_finite=c9_finite,
                passed=not reasons,
                failure_reasons=reasons,
            ),
        )
    return Exp466QualificationAntigravity(
        fold_results=tuple(fold_results),
        normal_trial_counts=MappingProxyType(
            {
                seed: FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed
                for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            },
        ),
        normal_confirmations=MappingProxyType(normal_confirmations),
        label_query_count=FROZEN_CONFIG_ANTIGRAVITY.label_query_count_must_remain,
        official_evaluate_count=(
            FROZEN_CONFIG_ANTIGRAVITY.official_evaluate_count_must_remain
        ),
        official_inspect_count=(
            FROZEN_CONFIG_ANTIGRAVITY.official_inspect_count_must_remain
        ),
        artifact_map_sha256=None,
    )


__all__ = ("qualify_antigravity",)
