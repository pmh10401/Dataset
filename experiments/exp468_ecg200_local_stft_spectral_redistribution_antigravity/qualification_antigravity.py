"""Exact ten-record raw-input qualification for Exp468 C10."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
    mann_whitney_auc_antigravity,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    Exp468FoldResultAntigravity,
    Exp468QualificationAntigravity,
)
from .local_stft_antigravity import score_local_stft_antigravity
from .predecessor_antigravity import (
    capture_exp465_authority_antigravity,
    score_c0_fold_antigravity,
)
from .scoring_antigravity import FoldMethodScoresAntigravity
from .synthetic_antigravity import primary_rows_antigravity


_METHODS_ANTIGRAVITY = ("C0", "S10", "C10")
_CHANGE_TYPES_ANTIGRAVITY = FROZEN_CONFIG_ANTIGRAVITY.change_types
_SCOPES_ANTIGRAVITY = ("normal",) + _CHANGE_TYPES_ANTIGRAVITY


def _readonly_indices_antigravity(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def blocked_fold_indices_antigravity() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the five frozen contiguous query blocks and their complements."""

    all_indices = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for query_source in np.array_split(all_indices, FROZEN_CONFIG_ANTIGRAVITY.folds):
        query = _readonly_indices_antigravity(query_source)
        fit = _readonly_indices_antigravity(
            np.setdiff1d(all_indices, query, assume_unique=True),
        )
        folds.append((fit, query))
    return tuple(folds)


def _validated_auc_tuple_antigravity(
    values: object,
    name: str,
) -> tuple[float, float, float, float]:
    if type(values) is not tuple or len(values) != len(_CHANGE_TYPES_ANTIGRAVITY):
        raise ValueError(f"{name} must be an exact four-value tuple")
    for index, value in enumerate(values):
        if type(value) is not float or not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{index}] must be an exact finite AUC float")
    return values


def _failure_reasons_antigravity(
    *,
    c0_per_type_auc: tuple[float, float, float, float],
    c10_per_type_auc: tuple[float, float, float, float],
    period_confirmations: int,
    normal_confirmations: int,
    all_scores_finite: bool,
) -> tuple[str, ...]:
    """Return every failed frozen fold gate in its preregistered order."""

    c0_auc = _validated_auc_tuple_antigravity(c0_per_type_auc, "c0_per_type_auc")
    c10_auc = _validated_auc_tuple_antigravity(c10_per_type_auc, "c10_per_type_auc")
    if type(period_confirmations) is not int or period_confirmations < 0:
        raise ValueError("period_confirmations must be a non-negative exact int")
    if type(normal_confirmations) is not int or normal_confirmations < 0:
        raise ValueError("normal_confirmations must be a non-negative exact int")
    if type(all_scores_finite) is not bool:
        raise ValueError("all_scores_finite must be exact bool")

    reasons: list[str] = []
    period_index = _CHANGE_TYPES_ANTIGRAVITY.index("period_change")
    if not all_scores_finite:
        reasons.append("nonfinite_scores")
    if c10_auc[period_index] < FROZEN_CONFIG_ANTIGRAVITY.period_auc_min:
        reasons.append("period_auc_below_0.85")
    if c10_auc[period_index] <= c0_auc[period_index]:
        reasons.append("period_auc_not_strictly_above_c0")
    if period_confirmations == 0:
        reasons.append("period_confirmation_missing")
    for index, change_type in enumerate(_CHANGE_TYPES_ANTIGRAVITY[:3]):
        if c10_auc[index] < c0_auc[index]:
            reasons.append(f"{change_type}_auc_below_c0")
    if normal_confirmations != 0:
        reasons.append("normal_confirmation_present")
    return tuple(reasons)


def _normalise_records_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
) -> tuple[tuple[tuple[int, int], FoldMethodScoresAntigravity], ...]:
    if not isinstance(records, Mapping):
        raise ValueError("records must be a mapping")
    expected_keys = tuple(
        (seed, fold_index)
        for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    if tuple(records) != expected_keys:
        raise ValueError("records must preserve exact frozen seed-major order")
    items = tuple(records.items())
    frozen_folds = blocked_fold_indices_antigravity()
    for key, record in items:
        if type(record) is not FoldMethodScoresAntigravity:
            raise ValueError("records must contain exact FoldMethodScoresAntigravity values")
        if key != (record.seed, record.fold_index):
            raise ValueError("record key differs from its seed/fold identity")
        if record.methods != _METHODS_ANTIGRAVITY or record.backend != "numpy":
            raise ValueError("record methods/backend differ from frozen Task 4 authority")
        expected_fit, expected_query = frozen_folds[record.fold_index]
        if (
            not np.array_equal(record.input_fit_indices, expected_fit)
            or not np.array_equal(record.input_query_indices, expected_query)
        ):
            raise ValueError("each record must preserve its exact blocked fold arrays")

    first_train_bytes = items[0][1].input_train140.tobytes(order="C")
    for _key, record in items:
        if (
            record.input_train140.shape
            != (
                FROZEN_CONFIG_ANTIGRAVITY.normal_rows,
                FROZEN_CONFIG_ANTIGRAVITY.interpolated_width,
            )
            or record.input_train140.tobytes(order="C") != first_train_bytes
        ):
            raise ValueError("all records must preserve the same exact 69x140 TRAIN bytes")

    expected_trials = np.arange(FROZEN_CONFIG_ANTIGRAVITY.normal_rows, dtype=np.int64)
    for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds:
        oof = np.concatenate(
            tuple(
                record.input_query_indices
                for identity, record in items
                if identity[0] == seed
            ),
        )
        if (
            len(oof) != len(expected_trials)
            or len(np.unique(oof)) != len(expected_trials)
            or not np.array_equal(oof, expected_trials)
        ):
            raise ValueError("each seed must cover exactly 69 blocked OOF trials once")
    return items


def _scope_rows_antigravity(record: FoldMethodScoresAntigravity) -> Mapping[str, np.ndarray]:
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
    authority: object,
) -> None:
    """Recompute C0, synthetic rows, raw S10, confirmation, rank, and C10."""

    c0 = score_c0_fold_antigravity(
        record.input_train140,
        record.input_fit_indices,
        record.input_query_indices,
        seed=record.seed,
        fold_index=record.fold_index,
        authority=authority,
    )
    if not np.array_equal(record.original["C0"], c0.normal_scores):
        raise ValueError("C0 input replay differs from stored normal scores")
    for change_type in _CHANGE_TYPES_ANTIGRAVITY:
        if not np.array_equal(
            record.primary["C0"][change_type],
            c0.primary_scores[change_type],
        ):
            raise ValueError("C0 input replay differs from stored primary scores")

    scope_rows = _scope_rows_antigravity(record)
    width = len(record.input_query_indices)
    combined_query = np.concatenate(
        tuple(scope_rows[scope] for scope in _SCOPES_ANTIGRAVITY),
        axis=0,
    )
    spectral = score_local_stft_antigravity(
        record.input_train140[record.input_fit_indices],
        combined_query,
        backend="numpy",
    )
    if spectral.backend != "numpy":
        raise ValueError("S10 input replay returned non-NumPy provenance")
    if (
        not np.array_equal(record.fit_raw_js, spectral.fit_raw_js)
        or record.fit_normal_max_raw_js != spectral.fit_normal_max_raw_js
    ):
        raise ValueError("S10 raw evidence differs from retained input replay")

    for scope_index, scope in enumerate(_SCOPES_ANTIGRAVITY):
        start = scope_index * width
        stop = start + width
        raw = spectral.query_raw_js[start:stop]
        rank = spectral.query_ranks[start:stop]
        confirmed = spectral.confirmation_mask[start:stop]
        stored_rank = (
            record.original["S10"]
            if scope == "normal"
            else record.primary["S10"][scope]
        )
        if (
            not np.array_equal(record.query_raw_js[scope], raw)
            or not np.array_equal(stored_rank, rank)
            or not np.array_equal(record.confirmed[scope], confirmed)
        ):
            raise ValueError("S10 raw evidence differs from retained input replay")
        stored_c0 = (
            record.original["C0"]
            if scope == "normal"
            else record.primary["C0"][scope]
        )
        stored_c10 = (
            record.original["C10"]
            if scope == "normal"
            else record.primary["C10"][scope]
        )
        expected_c10 = np.where(confirmed, np.maximum(stored_c0, rank), stored_c0)
        if not np.array_equal(stored_c10, expected_c10):
            raise ValueError("C10 score differs from raw S10/C0 input replay")


def _all_score_arrays_antigravity(
    record: FoldMethodScoresAntigravity,
) -> tuple[np.ndarray, ...]:
    return (
        record.fit_raw_js,
        *record.query_raw_js.values(),
        *record.original.values(),
        *(values for method in record.primary.values() for values in method.values()),
    )


def _method_auc_antigravity(
    record: FoldMethodScoresAntigravity,
    method: str,
) -> tuple[tuple[float, float, float, float], bool]:
    arrays = (
        record.original[method],
        *(record.primary[method][change_type] for change_type in _CHANGE_TYPES_ANTIGRAVITY),
    )
    finite = bool(all(np.isfinite(values).all() for values in arrays))
    if not finite:
        return (0.0, 0.0, 0.0, 0.0), False
    auc = tuple(
        float(
            mann_whitney_auc_antigravity(
                record.original[method],
                record.primary[method][change_type],
            ),
        )
        for change_type in _CHANGE_TYPES_ANTIGRAVITY
    )
    return auc, True  # type: ignore[return-value]


def qualify_antigravity(
    records: Mapping[tuple[int, int], FoldMethodScoresAntigravity],
) -> Exp468QualificationAntigravity:
    """Replay exactly ten seed-major records and emit scalar fold qualification."""

    items = _normalise_records_antigravity(records)
    authority = capture_exp465_authority_antigravity()
    fold_results: list[Exp468FoldResultAntigravity] = []
    normal_confirmations = {seed: 0 for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds}
    for _identity, record in items:
        _replay_record_antigravity(record, authority)
        c0_auc, c0_finite = _method_auc_antigravity(record, "C0")
        c10_auc, c10_finite = _method_auc_antigravity(record, "C10")
        all_scores_finite = bool(
            c0_finite
            and c10_finite
            and all(np.isfinite(values).all() for values in _all_score_arrays_antigravity(record))
        )
        period_confirmations = int(np.count_nonzero(record.confirmed["period_change"]))
        fold_normal_confirmations = int(np.count_nonzero(record.confirmed["normal"]))
        normal_confirmations[record.seed] += fold_normal_confirmations
        reasons = _failure_reasons_antigravity(
            c0_per_type_auc=c0_auc,
            c10_per_type_auc=c10_auc,
            period_confirmations=period_confirmations,
            normal_confirmations=fold_normal_confirmations,
            all_scores_finite=all_scores_finite,
        )
        fold_results.append(
            Exp468FoldResultAntigravity(
                seed=record.seed,
                fold_index=record.fold_index,
                c0_per_type_auc=c0_auc,
                c10_per_type_auc=c10_auc,
                period_confirmations=period_confirmations,
                normal_confirmations=fold_normal_confirmations,
                all_scores_finite=all_scores_finite,
                passed=not reasons,
                failure_reasons=reasons,
            ),
        )
    return Exp468QualificationAntigravity(
        fold_results=tuple(fold_results),
        normal_trial_counts=MappingProxyType(
            {
                seed: FROZEN_CONFIG_ANTIGRAVITY.normal_trials_per_seed
                for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            },
        ),
        normal_confirmations=MappingProxyType(normal_confirmations),
        label_query_count=0,
        artifact_map_sha256=None,
    )


__all__ = (
    "blocked_fold_indices_antigravity",
    "qualify_antigravity",
)
