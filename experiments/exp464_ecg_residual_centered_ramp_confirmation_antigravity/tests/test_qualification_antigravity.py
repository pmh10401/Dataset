from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from math import sqrt
import unittest
from unittest.mock import patch

import numpy as np


METHODS_ANTIGRAVITY = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
CONFIRMATION_TYPES_ANTIGRAVITY = ("normal",) + TYPES_ANTIGRAVITY
PAIR_TYPES_ANTIGRAVITY = ("gradual_drift", "period_change")
SEEDS_ANTIGRAVITY = (20260721, 20261721)
FOLDS_ANTIGRAVITY = tuple(range(5))


def train_antigravity() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 1413 * 140, dtype=np.float32).reshape(1413, 140)
    values.setflags(write=False)
    return values


def _vector_antigravity(values: tuple[float, ...]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def _auc_antigravity(normal: tuple[float, ...], change: tuple[float, ...]) -> float:
    normal_array = _vector_antigravity(normal)
    change_array = _vector_antigravity(change)
    wins = sum(
        int(change_value > normal_value)
        for change_value in change_array
        for normal_value in normal_array
    )
    ties = sum(
        int(change_value == normal_value)
        for change_value in change_array
        for normal_value in normal_array
    )
    return (wins + 0.5 * ties) / (len(change_array) * len(normal_array))


def _wilson_upper_antigravity(
    normal: tuple[float, ...],
    *,
    gate: float = 0.995,
) -> float:
    alerts = sum(score >= gate for score in normal)
    total = len(normal)
    z = 1.959963984540054
    proportion = alerts / total
    z_squared = z**2
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    adjustment = z * sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total
    )
    return min(1.0, (center + adjustment) / denominator)


def _mean_auc_antigravity(primary_scores: dict[str, tuple[float, ...]], normal: tuple[float, ...]) -> float:
    return sum(_auc_antigravity(normal, primary_scores[name]) for name in TYPES_ANTIGRAVITY) / len(
        TYPES_ANTIGRAVITY
    )


def _counts_antigravity(normal: tuple[float, ...], primary: tuple[float, ...]) -> tuple[int, int, int]:
    total = len(normal) * len(primary)
    fixed = 0
    broken = 0
    for change_score in primary:
        for normal_score in normal:
            correct_c0 = normal_score < normal_score
            del correct_c0
    return total, fixed, broken


def _fixed_broken_antigravity(
    c0_normal: tuple[float, ...],
    c0_change: tuple[float, ...],
    c6_normal: tuple[float, ...],
    c6_change: tuple[float, ...],
) -> tuple[int, int, int]:
    total = len(c0_normal) * len(c0_change)
    fixed = 0
    broken = 0
    for normal_index, c0_normal_score in enumerate(c0_normal):
        c6_normal_score = c6_normal[normal_index]
        for change_index, c0_change_score in enumerate(c0_change):
            c6_change_score = c6_change[change_index]
            if c0_change_score == c0_normal_score or c6_change_score == c6_normal_score:
                continue
            correct_c0 = c0_change_score > c0_normal_score
            correct_c6 = c6_change_score > c6_normal_score
            if (not correct_c0) and correct_c6:
                fixed += 1
            elif correct_c0 and (not correct_c6):
                broken += 1
    return total, fixed, broken


def _primary_scores_antigravity(
    *,
    level: tuple[float, ...] = (0.92, 0.94),
    gradual: tuple[float, ...] = (0.93, 0.97),
    amplitude: tuple[float, ...] = (0.91, 0.96),
    period: tuple[float, ...] = (0.94, 0.98),
) -> dict[str, tuple[float, ...]]:
    return {
        "level_shift": level,
        "gradual_drift": gradual,
        "amplitude_scaling": amplitude,
        "period_change": period,
    }


def _uniform_scores_antigravity(
    length: int,
    pair: tuple[float, float],
) -> tuple[float, ...]:
    midpoint = length // 2
    return tuple(pair[0] for _ in range(midpoint)) + tuple(
        pair[1] for _ in range(length - midpoint)
    )


def _length_matched_primary_scores_antigravity(
    length: int,
    *,
    level: tuple[float, float] = (1.10, 1.11),
    gradual: tuple[float, float] = (1.12, 1.13),
    amplitude: tuple[float, float] = (1.14, 1.15),
    period: tuple[float, float] = (1.16, 1.17),
) -> dict[str, tuple[float, ...]]:
    return {
        "level_shift": _uniform_scores_antigravity(length, level),
        "gradual_drift": _uniform_scores_antigravity(length, gradual),
        "amplitude_scaling": _uniform_scores_antigravity(length, amplitude),
        "period_change": _uniform_scores_antigravity(length, period),
    }


def _confidence_counts_antigravity(
    *,
    gradual: int = 2,
    period: int = 2,
    level: int = 2,
    amplitude: int = 2,
    normal: int = 1,
) -> dict[str, int]:
    return {
        "normal": normal,
        "level_shift": level,
        "gradual_drift": gradual,
        "amplitude_scaling": amplitude,
        "period_change": period,
    }


def _fold_container_antigravity(method_payloads: dict[str, dict[str, object]]):
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
    )

    return FoldMethodScoresAntigravity(
        methods=METHODS_ANTIGRAVITY,
        original={method: np.asarray(method_payloads[method]["normal"], dtype=np.float64) for method in METHODS_ANTIGRAVITY},
        primary={
            method: {
                change_type: np.asarray(method_payloads[method]["primary"][change_type], dtype=np.float64)
                for change_type in TYPES_ANTIGRAVITY
            }
            for method in METHODS_ANTIGRAVITY
        },
        confirmation_counts={
            method: dict(method_payloads[method]["confirmation_counts"])
            for method in METHODS_ANTIGRAVITY
        },
        fit_amplitudes={
            method: np.asarray(
                method_payloads[method].get("fit_amplitudes", ()),
                dtype=np.float64,
            )
            for method in METHODS_ANTIGRAVITY
        },
        query_amplitudes={
            method: {
                scope: np.asarray(values, dtype=np.float64)
                for scope, values in method_payloads[method].get(
                    "query_amplitudes", {}
                ).items()
            }
            for method in METHODS_ANTIGRAVITY
        },
        query_cosines={
            method: {
                scope: np.asarray(values, dtype=np.float64)
                for scope, values in method_payloads[method].get(
                    "query_cosines", {}
                ).items()
            }
            for method in METHODS_ANTIGRAVITY
        },
        raw_evidence_kind={
            method: str(method_payloads[method].get("raw_evidence_kind", ""))
            for method in METHODS_ANTIGRAVITY
        },
        raw_evidence_source_method={
            method: str(method_payloads[method].get("raw_evidence_source_method", ""))
            for method in METHODS_ANTIGRAVITY
        },
        raw_fit_vector_blobs={
            method: str(method_payloads[method].get("raw_fit_vector_blob", ""))
            for method in METHODS_ANTIGRAVITY
        },
        raw_query_vector_blobs={
            method: {
                scope: str(blob)
                for scope, blob in method_payloads[method].get(
                    "raw_query_vector_blobs", {}
                ).items()
            }
            for method in METHODS_ANTIGRAVITY
        },
        feature_dimensions={method: int(method_payloads[method].get("feature_dimension", 1)) for method in METHODS_ANTIGRAVITY},
        provenance={method: str(method_payloads[method].get("provenance", f"fixture_{method.lower()}")) for method in METHODS_ANTIGRAVITY},
    )


def _rank_query_amplitudes_antigravity(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((value * 100.0) + 1.5 for value in values)


def _evidence_payload_antigravity(
    *,
    normal: tuple[float, ...],
    primary: dict[str, tuple[float, ...]],
    normal_confirmed: int,
    primary_confirmed: int,
    raw_evidence_kind: str,
) -> dict[str, object]:
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.raw_evidence_antigravity import (
        RAW_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY,
        encode_vector_matrix_antigravity,
    )

    if raw_evidence_kind not in (
        RAW_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY,
    ):
        raise ValueError("raw_evidence_kind is invalid")

    fit_amplitudes = tuple(float(index) for index in range(1, 100))
    normal_cosines = tuple(
        1.0 if index < normal_confirmed else 0.94 for index in range(len(normal))
    )
    primary_cosines = tuple(
        1.0 if index < primary_confirmed else 0.94 for index in range(len(normal))
    )
    query_amplitudes = {
        "normal": _rank_query_amplitudes_antigravity(normal),
        **{
            change_type: _rank_query_amplitudes_antigravity(primary[change_type])
            for change_type in TYPES_ANTIGRAVITY
        },
    }
    query_cosines = {
        "normal": normal_cosines,
        **{change_type: primary_cosines for change_type in TYPES_ANTIGRAVITY},
    }
    return {
        "fit_amplitudes": fit_amplitudes,
        "query_amplitudes": query_amplitudes,
        "query_cosines": query_cosines,
        "raw_evidence_kind": raw_evidence_kind,
        "raw_fit_vector_blob": encode_vector_matrix_antigravity(
            _raw_matrix_antigravity(
                amplitudes=fit_amplitudes,
                cosines=tuple(1.0 for _ in fit_amplitudes),
                raw_evidence_kind=raw_evidence_kind,
            )
        ),
        "raw_query_vector_blobs": {
            scope: encode_vector_matrix_antigravity(
                _raw_matrix_antigravity(
                    amplitudes=query_amplitudes[scope],
                    cosines=query_cosines[scope],
                    raw_evidence_kind=raw_evidence_kind,
                )
            )
            for scope in CONFIRMATION_TYPES_ANTIGRAVITY
        },
    }


def _raw_basis_antigravity(raw_evidence_kind: str) -> np.ndarray:
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
        centered_suffix_basis_antigravity,
    )
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.config_antigravity import (
        FROZEN_CONFIG_ANTIGRAVITY,
    )
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.raw_evidence_antigravity import (
        RAW_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY,
    )

    if raw_evidence_kind in (
        RAW_CENTERED_ANTIGRAVITY,
        TEMPLATE_RESIDUAL_CENTERED_ANTIGRAVITY,
    ):
        return np.asarray(centered_suffix_basis_antigravity(), dtype=np.float64)
    if raw_evidence_kind == TEMPLATE_RESIDUAL_UNCENTERED_ANTIGRAVITY:
        values = np.linspace(
            0.0,
            1.0,
            FROZEN_CONFIG_ANTIGRAVITY.normal2_width - FROZEN_CONFIG_ANTIGRAVITY.onset,
            dtype=np.float64,
        )
        return values / float(np.linalg.norm(values))
    raise ValueError("raw_evidence_kind is invalid")


@lru_cache(maxsize=None)
def _orthogonal_unit_antigravity(raw_evidence_kind: str) -> tuple[float, ...]:
    basis = _raw_basis_antigravity(raw_evidence_kind)
    probe = np.zeros_like(basis)
    probe[0] = 1.0
    orthogonal = probe - (float(np.dot(probe, basis)) * basis)
    norm = float(np.linalg.norm(orthogonal))
    if norm <= 0.0:
        probe = np.arange(1.0, float(len(basis)) + 1.0, dtype=np.float64)
        orthogonal = probe - (float(np.dot(probe, basis)) * basis)
        norm = float(np.linalg.norm(orthogonal))
    if norm <= 0.0:
        raise ValueError("could not construct orthogonal basis")
    return tuple(float(value) for value in (orthogonal / norm))


def _raw_matrix_antigravity(
    *,
    amplitudes: tuple[float, ...],
    cosines: tuple[float, ...],
    raw_evidence_kind: str,
) -> np.ndarray:
    if len(amplitudes) != len(cosines):
        raise ValueError("amplitude/cosine lengths must match")
    basis = _raw_basis_antigravity(raw_evidence_kind)
    orthogonal = np.asarray(
        _orthogonal_unit_antigravity(raw_evidence_kind),
        dtype=np.float64,
    )
    rows: list[np.ndarray] = []
    for amplitude, cosine in zip(amplitudes, cosines, strict=True):
        if not -1.0 <= cosine <= 1.0:
            raise ValueError("cosine is out of range")
        if amplitude == 0.0:
            if abs(cosine) > 1e-12:
                raise ValueError("zero amplitude cannot carry a non-zero cosine")
            rows.append(np.zeros_like(basis))
            continue
        if abs(cosine) < 1e-12:
            raise ValueError("non-zero amplitude cannot carry a zero cosine")
        norm = abs(amplitude / cosine)
        orthogonal_scale = sqrt(max((norm * norm) - (amplitude * amplitude), 0.0))
        rows.append((amplitude * basis) + (orthogonal_scale * orthogonal))
    return np.asarray(rows, dtype=np.float64)


def _scores_from_evidence_antigravity(
    *,
    baseline: tuple[float, ...],
    fit_amplitudes: tuple[float, ...],
    query_amplitudes: tuple[float, ...],
    query_cosines: tuple[float, ...] | None,
    rank_only: bool,
    always_max: bool,
) -> tuple[float, ...]:
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
        anomaly_rank_antigravity,
    )

    ranks = anomaly_rank_antigravity(
        np.asarray(fit_amplitudes, dtype=np.float64),
        np.asarray(query_amplitudes, dtype=np.float64),
    )
    if rank_only:
        return tuple(float(value) for value in ranks)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if always_max:
        return tuple(float(value) for value in np.maximum(baseline_array, ranks))
    assert query_cosines is not None
    confirmed = np.asarray(query_cosines, dtype=np.float64) >= 0.95
    return tuple(
        float(value)
        for value in np.where(
            confirmed,
            np.maximum(baseline_array, ranks),
            baseline_array,
        )
    )


def _strip_raw_blobs_antigravity(evidence: dict[str, object]) -> dict[str, object]:
    return {
        "fit_amplitudes": evidence["fit_amplitudes"],
        "query_amplitudes": evidence["query_amplitudes"],
        "query_cosines": evidence["query_cosines"],
    }


@lru_cache(maxsize=1)
def _exact_control_rows_antigravity() -> dict[tuple[str, int, int], dict[str, object]]:
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.control_row_evidence_antigravity import (
        load_control_row_evidence_antigravity,
    )

    rows: dict[tuple[str, int, int], dict[str, object]] = {}
    for row in load_control_row_evidence_antigravity().rows:
        method = row["method"]
        seed = row["seed"]
        fold_index = row["fold_index"]
        assert isinstance(method, str)
        assert isinstance(seed, int)
        assert isinstance(fold_index, int)
        rows[(method, seed, fold_index)] = {
            "normal": tuple(row["normal_scores"]),
            "primary": {
                change_type: tuple(row["primary_scores"][change_type])
                for change_type in TYPES_ANTIGRAVITY
            },
            "confirmation_counts": _confidence_counts_antigravity(
                normal=0, gradual=0, period=0, level=0, amplitude=0
            ),
            "provenance": f"tracked_{row['source_method'].lower()}_control",
        }
    return rows


def synthetic_fold_payload_antigravity(
    seed: int,
    fold_index: int,
) -> dict[str, dict[str, object]]:
    exact_rows = _exact_control_rows_antigravity()
    controls = {
        method: exact_rows[(method, seed, fold_index)]
        for method in ("C0", "C1", "C2", "C7")
    }
    c0_normal = controls["C0"]["normal"]
    c0_primary = controls["C0"]["primary"]
    c6_gradual = (1.20, 1.21)
    c6_period = (1.22, 1.23)
    if fold_index == 3:
        c6_period = (1.22, 1.23)
    if fold_index == 4:
        c6_period = (1.22, 1.23)
    c6_primary = _length_matched_primary_scores_antigravity(
        len(c0_normal),
        gradual=c6_gradual,
        period=c6_period,
    )
    c5_primary_seed = _length_matched_primary_scores_antigravity(
        len(c0_normal),
        period=(1.18, 1.19),
    )
    c8_primary_seed = _length_matched_primary_scores_antigravity(
        len(c0_normal),
        period=(1.06, 1.07),
    )
    c5_evidence = _evidence_payload_antigravity(
        normal=c0_normal,
        primary=c5_primary_seed,
        normal_confirmed=1,
        primary_confirmed=2,
        raw_evidence_kind="raw_centered",
    )
    c6_evidence = _evidence_payload_antigravity(
        normal=c0_normal,
        primary=c6_primary,
        normal_confirmed=1,
        primary_confirmed=len(c0_normal),
        raw_evidence_kind="template_residual_centered",
    )
    c8_evidence = _evidence_payload_antigravity(
        normal=c0_normal,
        primary=c8_primary_seed,
        normal_confirmed=1,
        primary_confirmed=2,
        raw_evidence_kind="template_residual_uncentered",
    )
    rows = {
        "C0": controls["C0"],
        "C1": controls["C1"],
        "C2": controls["C2"],
        "C3": {
            "normal": _scores_from_evidence_antigravity(
                baseline=c0_normal,
                fit_amplitudes=c6_evidence["fit_amplitudes"],
                query_amplitudes=c6_evidence["query_amplitudes"]["normal"],
                query_cosines=None,
                rank_only=True,
                always_max=False,
            ),
            "primary": {
                change_type: _scores_from_evidence_antigravity(
                    baseline=c0_primary[change_type],
                    fit_amplitudes=c6_evidence["fit_amplitudes"],
                    query_amplitudes=c6_evidence["query_amplitudes"][change_type],
                    query_cosines=None,
                    rank_only=True,
                    always_max=False,
                )
                for change_type in TYPES_ANTIGRAVITY
            },
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "rank_only",
            "raw_evidence_source_method": "C6",
            **_strip_raw_blobs_antigravity(c6_evidence),
        },
        "C4": {
            "normal": _scores_from_evidence_antigravity(
                baseline=c0_normal,
                fit_amplitudes=c6_evidence["fit_amplitudes"],
                query_amplitudes=c6_evidence["query_amplitudes"]["normal"],
                query_cosines=None,
                rank_only=False,
                always_max=True,
            ),
            "primary": {
                change_type: _scores_from_evidence_antigravity(
                    baseline=c0_primary[change_type],
                    fit_amplitudes=c6_evidence["fit_amplitudes"],
                    query_amplitudes=c6_evidence["query_amplitudes"][change_type],
                    query_cosines=None,
                    rank_only=False,
                    always_max=True,
                )
                for change_type in TYPES_ANTIGRAVITY
            },
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "always_max",
            "raw_evidence_source_method": "C6",
            **_strip_raw_blobs_antigravity(c6_evidence),
        },
        "C5": {
            "normal": _scores_from_evidence_antigravity(
                baseline=c0_normal,
                fit_amplitudes=c5_evidence["fit_amplitudes"],
                query_amplitudes=c5_evidence["query_amplitudes"]["normal"],
                query_cosines=c5_evidence["query_cosines"]["normal"],
                rank_only=False,
                always_max=False,
            ),
            "primary": {
                change_type: _scores_from_evidence_antigravity(
                    baseline=c0_primary[change_type],
                    fit_amplitudes=c5_evidence["fit_amplitudes"],
                    query_amplitudes=c5_evidence["query_amplitudes"][change_type],
                    query_cosines=c5_evidence["query_cosines"][change_type],
                    rank_only=False,
                    always_max=False,
                )
                for change_type in TYPES_ANTIGRAVITY
            },
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "raw_centered",
            "raw_evidence_source_method": "C5",
            **c5_evidence,
        },
        "C6": {
            "normal": _scores_from_evidence_antigravity(
                baseline=c0_normal,
                fit_amplitudes=c6_evidence["fit_amplitudes"],
                query_amplitudes=c6_evidence["query_amplitudes"]["normal"],
                query_cosines=c6_evidence["query_cosines"]["normal"],
                rank_only=False,
                always_max=False,
            ),
            "primary": c6_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "template_residual_centered",
            "raw_evidence_source_method": "C6",
            **{
                **c6_evidence,
                "primary": {
                    change_type: _scores_from_evidence_antigravity(
                        baseline=c0_primary[change_type],
                        fit_amplitudes=c6_evidence["fit_amplitudes"],
                        query_amplitudes=c6_evidence["query_amplitudes"][change_type],
                        query_cosines=c6_evidence["query_cosines"][change_type],
                        rank_only=False,
                        always_max=False,
                    )
                    for change_type in TYPES_ANTIGRAVITY
                },
            },
        },
        "C7": controls["C7"],
        "C8": {
            "normal": _scores_from_evidence_antigravity(
                baseline=c0_normal,
                fit_amplitudes=c8_evidence["fit_amplitudes"],
                query_amplitudes=c8_evidence["query_amplitudes"]["normal"],
                query_cosines=c8_evidence["query_cosines"]["normal"],
                rank_only=False,
                always_max=False,
            ),
            "primary": {
                change_type: _scores_from_evidence_antigravity(
                    baseline=c0_primary[change_type],
                    fit_amplitudes=c8_evidence["fit_amplitudes"],
                    query_amplitudes=c8_evidence["query_amplitudes"][change_type],
                    query_cosines=c8_evidence["query_cosines"][change_type],
                    rank_only=False,
                    always_max=False,
                )
                for change_type in TYPES_ANTIGRAVITY
            },
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "uncentered",
            "raw_evidence_source_method": "C8",
            **c8_evidence,
        },
    }
    return rows


def fold_scorer_antigravity(
    _train: np.ndarray,
    _fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
    mutated_payloads: dict[tuple[int, int], dict[str, dict[str, object]]] | None = None,
):
    del query_indices
    payloads = synthetic_fold_payload_antigravity(seed, fold_index)
    if mutated_payloads is not None and (seed, fold_index) in mutated_payloads:
        payloads = mutated_payloads[(seed, fold_index)]
    return _fold_container_antigravity(payloads)


def qualification_antigravity(
    mutated_payloads: dict[tuple[int, int], dict[str, dict[str, object]]] | None = None,
):
    if mutated_payloads is not None:
        raise ValueError("mutated_payloads is no longer supported for qualification fixtures")
    return _source_free_qualification_antigravity()


@lru_cache(maxsize=1)
def _source_free_qualification_antigravity():
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
        capture_exp463_authority_antigravity,
    )
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    with (
        patch(
            "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.sources_antigravity.load_normal2_train_antigravity",
            side_effect=AssertionError("focused Task4 fixtures must not load Exp464 TRAIN"),
        ),
        patch(
            "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity.score_fold_methods_antigravity",
            side_effect=AssertionError("focused Task4 fixtures must not call the production fold scorer"),
        ),
    ):
        return _qualify_methods_with_dependencies_antigravity(
            authority=capture_exp463_authority_antigravity(),
            loader=train_antigravity,
            fold_scorer=fold_scorer_antigravity,
        )


def _boundary_pass_row_antigravity():
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
        Exp464MethodFoldMetricAntigravity,
    )

    normal = tuple(float(index) / 100.0 for index in range(100))

    evidence = _evidence_payload_antigravity(
        normal=normal,
        primary={
            "level_shift": _auc_boundary_scores_antigravity(85),
            "gradual_drift": _auc_boundary_scores_antigravity(85),
            "amplitude_scaling": _auc_boundary_scores_antigravity(90),
            "period_change": _auc_boundary_scores_antigravity(100),
        },
        normal_confirmed=100,
        primary_confirmed=100,
        raw_evidence_kind="template_residual_centered",
    )

    return Exp464MethodFoldMetricAntigravity(
        method="C6",
        seed=20260721,
        fold_index=3,
        normal_trials=100,
        normal_alerts=99,
        normal_wilson_upper=1.0,
        per_type_auc={name: 0.0 for name in TYPES_ANTIGRAVITY},
        mean_auc=0.0,
        scores_finite=False,
        normal_scores=normal,
        primary_scores={
            "level_shift": _auc_boundary_scores_antigravity(85),
            "gradual_drift": _auc_boundary_scores_antigravity(85),
            "amplitude_scaling": _auc_boundary_scores_antigravity(90),
            "period_change": _auc_boundary_scores_antigravity(100),
        },
        confirmation_counts=_confidence_counts_antigravity(),
        confirmation_rates={name: 0.0 for name in CONFIRMATION_TYPES_ANTIGRAVITY},
        pair_diagnostics={
            change_type: {"total_pairs": 0, "fixed_pairs": 0, "broken_pairs": 0}
            for change_type in PAIR_TYPES_ANTIGRAVITY
        },
        passed=False,
        failure_reasons=(),
        feature_dimension=2,
        provenance="boundary_fixture",
        fit_amplitudes=evidence["fit_amplitudes"],
        query_amplitudes=evidence["query_amplitudes"],
        query_cosines=evidence["query_cosines"],
        raw_evidence_kind="template_residual_centered",
        raw_evidence_source_method="C6",
        raw_fit_vector_blob=evidence["raw_fit_vector_blob"],
        raw_query_vector_blobs=evidence["raw_query_vector_blobs"],
    )


def _auc_boundary_scores_antigravity(wins: int) -> tuple[float, ...]:
    if type(wins) is not int or not 0 <= wins <= 100:
        raise ValueError("wins must be an integer in [0, 100]")
    return tuple([-1.0] * (100 - wins) + [1.5] * wins)


def _c6_row_evidence_antigravity(
    normal_scores: tuple[float, ...],
    primary_scores: dict[str, tuple[float, ...]],
    *,
    normal_confirmed: int,
    primary_confirmed: int,
) -> dict[str, object]:
    evidence = _evidence_payload_antigravity(
        normal=normal_scores,
        primary=primary_scores,
        normal_confirmed=normal_confirmed,
        primary_confirmed=primary_confirmed,
        raw_evidence_kind="template_residual_centered",
    )
    return {
        "fit_amplitudes": evidence["fit_amplitudes"],
        "query_amplitudes": evidence["query_amplitudes"],
        "query_cosines": evidence["query_cosines"],
        "raw_evidence_kind": "template_residual_centered",
        "raw_evidence_source_method": "C6",
        "raw_fit_vector_blob": evidence["raw_fit_vector_blob"],
        "raw_query_vector_blobs": evidence["raw_query_vector_blobs"],
    }


def _replace_c6_row_with_numeric_evidence_antigravity(
    qualification,
    *,
    seed: int,
    fold_index: int,
    promoted_indices: dict[str, tuple[int, ...]],
):
    """Return a contract whose C6 row is derived from real raw numeric evidence."""
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.raw_evidence_antigravity import (
        encode_vector_matrix_antigravity,
    )

    rows = list(qualification.fold_metrics)
    c0 = next(
        row
        for row in rows
        if row.method == "C0" and row.seed == seed and row.fold_index == fold_index
    )
    c6_index = next(
        index
        for index, row in enumerate(rows)
        if row.method == "C6" and row.seed == seed and row.fold_index == fold_index
    )
    c6 = rows[c6_index]
    baselines = {"normal": c0.normal_scores, **dict(c0.primary_scores)}
    query_amplitudes: dict[str, tuple[float, ...]] = {}
    query_cosines: dict[str, tuple[float, ...]] = {}
    raw_query_vector_blobs: dict[str, str] = {}
    scores: dict[str, tuple[float, ...]] = {}
    counts: dict[str, int] = {}
    for scope in CONFIRMATION_TYPES_ANTIGRAVITY:
        promoted = frozenset(promoted_indices.get(scope, ()))
        if any(type(index) is not int or not 0 <= index < len(baselines[scope]) for index in promoted):
            raise ValueError(f"promoted index is out of range for {scope}")
        amplitudes = tuple(
            200.0 if index in promoted else 0.5
            for index in range(len(baselines[scope]))
        )
        cosines = tuple(
            1.0 if index in promoted else 0.94
            for index in range(len(baselines[scope]))
        )
        query_amplitudes[scope] = amplitudes
        query_cosines[scope] = cosines
        raw_query_vector_blobs[scope] = encode_vector_matrix_antigravity(
            _raw_matrix_antigravity(
                amplitudes=amplitudes,
                cosines=cosines,
                raw_evidence_kind="template_residual_centered",
            )
        )
        scores[scope] = _scores_from_evidence_antigravity(
            baseline=baselines[scope],
            fit_amplitudes=c6.fit_amplitudes,
            query_amplitudes=amplitudes,
            query_cosines=cosines,
            rank_only=False,
            always_max=False,
        )
        counts[scope] = len(promoted)
    rows[c6_index] = replace(
        c6,
        normal_scores=scores["normal"],
        primary_scores={scope: scores[scope] for scope in TYPES_ANTIGRAVITY},
        confirmation_counts=counts,
        confirmation_rates={scope: 0.0 for scope in CONFIRMATION_TYPES_ANTIGRAVITY},
        passed=True,
        failure_reasons=(),
        query_amplitudes=query_amplitudes,
        query_cosines=query_cosines,
        raw_query_vector_blobs=raw_query_vector_blobs,
    )
    for method, rank_only, always_max in (
        ("C3", True, False),
        ("C4", False, True),
    ):
        index = next(
            index
            for index, row in enumerate(rows)
            if row.method == method and row.seed == seed and row.fold_index == fold_index
        )
        row = rows[index]
        rows[index] = replace(
            row,
            normal_scores=_scores_from_evidence_antigravity(
                baseline=c0.normal_scores,
                fit_amplitudes=c6.fit_amplitudes,
                query_amplitudes=query_amplitudes["normal"],
                query_cosines=None,
                rank_only=rank_only,
                always_max=always_max,
            ),
            primary_scores={
                scope: _scores_from_evidence_antigravity(
                    baseline=c0.primary_scores[scope],
                    fit_amplitudes=c6.fit_amplitudes,
                    query_amplitudes=query_amplitudes[scope],
                    query_cosines=None,
                    rank_only=rank_only,
                    always_max=always_max,
                )
                for scope in TYPES_ANTIGRAVITY
            },
            fit_amplitudes=c6.fit_amplitudes,
            query_amplitudes=query_amplitudes,
            query_cosines=query_cosines,
        )
    return replace(qualification, fold_metrics=tuple(rows))


def numeric_boundary_qualification_antigravity():
    """Build the minimal C0-fallback C6 contract that passes every frozen gate."""
    qualification = qualification_antigravity()
    for seed in SEEDS_ANTIGRAVITY:
        for fold_index in FOLDS_ANTIGRAVITY:
            qualification = _replace_c6_row_with_numeric_evidence_antigravity(
                qualification,
                seed=seed,
                fold_index=fold_index,
                promoted_indices={},
            )
    return _replace_c6_row_with_numeric_evidence_antigravity(
        qualification,
        seed=20261721,
        fold_index=3,
        promoted_indices={"gradual_drift": tuple(range(23))},
    )


class QualificationAntigravityTest(unittest.TestCase):
    def test_focused_qualification_fixtures_never_touch_train_loader_or_production_scorer_antigravity(self) -> None:
        """Break caught: focused Task4 tests silently re-enter live Exp464/Exp463 qualification."""
        result = qualification_antigravity()
        self.assertEqual(len(result.fold_metrics), 90)

    def test_qualification_emits_exact_ordered_rows_and_summaries_antigravity(self) -> None:
        """Break caught: a method, seed, fold, confirmation field, or summary is dropped."""
        result = qualification_antigravity()
        self.assertEqual(result.method_order, METHODS_ANTIGRAVITY)
        self.assertEqual(len(result.fold_metrics), 90)
        self.assertEqual(
            tuple((row.method, row.seed, row.fold_index) for row in result.fold_metrics),
            tuple(
                (method, seed, fold)
                for method in METHODS_ANTIGRAVITY
                for seed in SEEDS_ANTIGRAVITY
                for fold in FOLDS_ANTIGRAVITY
            ),
        )
        self.assertEqual(tuple(summary.method for summary in result.summaries), METHODS_ANTIGRAVITY)
        candidate_rows = [
            row for row in result.fold_metrics if row.method == "C6"
        ]
        self.assertEqual(len(candidate_rows), 10)
        for row in candidate_rows:
            self.assertEqual(tuple(row.per_type_auc), TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.primary_scores), TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.confirmation_counts), CONFIRMATION_TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.confirmation_rates), CONFIRMATION_TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.pair_diagnostics), PAIR_TYPES_ANTIGRAVITY)
            self.assertTrue(row.scores_finite)

    def test_metric_recomputation_is_independent_from_score_evidence_and_boundaries_antigravity(self) -> None:
        """Break caught: stored AUC, mean, Wilson, finite, or pass flags override scores."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            Exp464MethodFoldMetricAntigravity,
            recompute_metric_from_scores_antigravity,
        )

        normal = (0.10, 0.20)
        primary_scores = _primary_scores_antigravity(
            level=(0.10, 0.20),
            gradual=(0.85, 0.95),
            amplitude=(0.20, 0.90),
            period=(0.907392736783864, 0.95),
        )
        evidence = _c6_row_evidence_antigravity(
            normal,
            primary_scores,
            normal_confirmed=0,
            primary_confirmed=2,
        )
        row = Exp464MethodFoldMetricAntigravity(
            method="C6",
            seed=20260721,
            fold_index=3,
            normal_trials=2,
            normal_alerts=99,
            normal_wilson_upper=0.999,
            per_type_auc={name: 0.0 for name in TYPES_ANTIGRAVITY},
            mean_auc=0.0,
            scores_finite=False,
            normal_scores=normal,
            primary_scores=primary_scores,
            confirmation_counts=_confidence_counts_antigravity(),
            confirmation_rates={name: 0.0 for name in CONFIRMATION_TYPES_ANTIGRAVITY},
            pair_diagnostics={
                change_type: {"total_pairs": 0, "fixed_pairs": 0, "broken_pairs": 0}
                for change_type in PAIR_TYPES_ANTIGRAVITY
            },
            passed=False,
            failure_reasons=(),
            feature_dimension=2,
            provenance="fixture",
            fit_amplitudes=evidence["fit_amplitudes"],
            query_amplitudes=evidence["query_amplitudes"],
            query_cosines=evidence["query_cosines"],
            raw_evidence_kind=evidence["raw_evidence_kind"],
            raw_evidence_source_method=evidence["raw_evidence_source_method"],
            raw_fit_vector_blob=evidence["raw_fit_vector_blob"],
            raw_query_vector_blobs=evidence["raw_query_vector_blobs"],
        )
        recomputed = recompute_metric_from_scores_antigravity(row)
        expected_auc = {
            name: _auc_antigravity(normal, primary_scores[name]) for name in TYPES_ANTIGRAVITY
        }
        expected_mean = sum(expected_auc.values()) / len(expected_auc)
        self.assertEqual(dict(recomputed.per_type_auc), expected_auc)
        self.assertEqual(recomputed.mean_auc, expected_mean)
        self.assertEqual(recomputed.normal_alerts, 0)
        self.assertAlmostEqual(
            recomputed.normal_wilson_upper,
            _wilson_upper_antigravity(normal),
            places=15,
        )
        self.assertTrue(recomputed.scores_finite)
        self.assertIn("level_shift_auc_below_min", recomputed.failure_reasons)
        self.assertNotIn("historical_period_change_below_min:20260721:3", recomputed.failure_reasons)

        nonfinite = replace(
            row,
            normal_scores=(float("nan"), 0.20),
            primary_scores={name: (float("nan"), 0.20) for name in TYPES_ANTIGRAVITY},
        )
        nonfinite_recomputed = recompute_metric_from_scores_antigravity(nonfinite)
        self.assertFalse(nonfinite_recomputed.scores_finite)
        self.assertEqual(
            dict(nonfinite_recomputed.per_type_auc),
            {name: 0.0 for name in TYPES_ANTIGRAVITY},
        )

    def test_boundary_row_hits_exact_auc_mean_and_wilson_edges_antigravity(self) -> None:
        """Break caught: score-derived row gates drift or stop emitting exact reasons."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            recompute_metric_from_scores_antigravity,
        )

        passing = recompute_metric_from_scores_antigravity(_boundary_pass_row_antigravity())
        self.assertEqual(passing.per_type_auc["level_shift"], 0.85)
        self.assertEqual(passing.per_type_auc["gradual_drift"], 0.85)
        self.assertEqual(passing.per_type_auc["amplitude_scaling"], 0.9)
        self.assertEqual(passing.per_type_auc["period_change"], 1.0)
        self.assertEqual(passing.mean_auc, 0.9)
        self.assertLessEqual(passing.normal_wilson_upper, 0.05)
        self.assertTrue(passing.scores_finite)
        self.assertEqual(passing.failure_reasons, ())

        per_type_cases = {
            change_type: {
                scope: _auc_boundary_scores_antigravity(
                    84 if scope == change_type else 100
                )
                for scope in TYPES_ANTIGRAVITY
            }
            for change_type in TYPES_ANTIGRAVITY
        }
        for change_type, scores in per_type_cases.items():
            with self.subTest(change_type=change_type):
                failed = recompute_metric_from_scores_antigravity(
                    replace(_boundary_pass_row_antigravity(), primary_scores=scores)
                )
                self.assertEqual(failed.per_type_auc[change_type], 0.84)
                self.assertEqual(
                    failed.failure_reasons,
                    (f"{change_type}_auc_below_min",),
                )

        mean_failed = recompute_metric_from_scores_antigravity(
            replace(
                _boundary_pass_row_antigravity(),
                primary_scores={
                    "level_shift": _auc_boundary_scores_antigravity(85),
                    "gradual_drift": _auc_boundary_scores_antigravity(85),
                    "amplitude_scaling": _auc_boundary_scores_antigravity(89),
                    "period_change": _auc_boundary_scores_antigravity(100),
                },
            )
        )
        self.assertEqual(mean_failed.mean_auc, 0.8975)
        self.assertEqual(mean_failed.failure_reasons, ("mean_auc_below_min",))

        wilson_failed = recompute_metric_from_scores_antigravity(
            replace(
                _boundary_pass_row_antigravity(),
                normal_scores=(1.01,) + _boundary_pass_row_antigravity().normal_scores[1:],
            )
        )
        self.assertGreater(wilson_failed.normal_wilson_upper, 0.05)
        self.assertEqual(
            wilson_failed.failure_reasons,
            ("normal_wilson_upper_above_max",),
        )

        nonfinite = recompute_metric_from_scores_antigravity(
            replace(
                _boundary_pass_row_antigravity(),
                normal_scores=(float("nan"),)
                + _boundary_pass_row_antigravity().normal_scores[1:],
            )
        )
        self.assertEqual(
            nonfinite.failure_reasons,
            (
                "nonfinite_scores",
                "normal_wilson_upper_above_max",
                "mean_auc_below_min",
                "level_shift_auc_below_min",
                "gradual_drift_auc_below_min",
                "amplitude_scaling_auc_below_min",
                "period_change_auc_below_min",
            ),
        )

    def test_normalization_rejects_wrong_order_identity_and_forged_c6_scores_antigravity(self) -> None:
        """Break caught: reordered rows or forged evidence-bearing C6 scores survive normalization."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            normalize_qualification_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        c6_index = next(
            index
            for index, row in enumerate(rows)
            if row.method == "C6" and row.seed == 20260721 and row.fold_index == 3
        )
        rows[c6_index] = replace(
            rows[c6_index],
            primary_scores={
                **rows[c6_index].primary_scores,
                "period_change": rows[c6_index].normal_scores,
            },
            passed=True,
            failure_reasons=(),
        )
        with self.assertRaises(ValueError):
            normalize_qualification_antigravity(
                replace(qualification, fold_metrics=tuple(rows)),
                authority,
            )

        reordered = list(qualification.fold_metrics)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(ValueError):
            normalize_qualification_antigravity(
                replace(qualification, fold_metrics=tuple(reordered)),
                authority,
            )

    def test_normalization_rejects_non_extremal_c0_mutation_before_c6_comparison_antigravity(self) -> None:
        """Break caught: non-extremal C0 row drift can no longer survive summary preservation."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            normalize_qualification_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        c0_index = next(
            index
            for index, row in enumerate(rows)
            if row.method == "C0" and row.seed == 20260721 and row.fold_index == 0
        )
        row = rows[c0_index]
        rows[c0_index] = replace(
            row,
            primary_scores={
                **row.primary_scores,
                "level_shift": tuple(score + 0.001 for score in row.primary_scores["level_shift"]),
            },
        )
        with self.assertRaises(ValueError):
            normalize_qualification_antigravity(
                replace(qualification, fold_metrics=tuple(rows)),
                authority,
            )

    def test_pair_diagnostics_use_all_ordered_pairs_and_strict_greater_than_only_antigravity(self) -> None:
        """Break caught: diagnostics skip ties, unordered pairs, or alter the decision contract."""
        qualification = qualification_antigravity()
        candidate = next(
            row
            for row in qualification.fold_metrics
            if row.method == "C6" and row.seed == 20260721 and row.fold_index == 0
        )
        baseline = next(
            row
            for row in qualification.fold_metrics
            if row.method == "C0" and row.seed == 20260721 and row.fold_index == 0
        )
        expected_total, expected_fixed, expected_broken = _fixed_broken_antigravity(
            baseline.normal_scores,
            baseline.primary_scores["gradual_drift"],
            candidate.normal_scores,
            candidate.primary_scores["gradual_drift"],
        )
        gradual = candidate.pair_diagnostics["gradual_drift"]
        self.assertEqual(gradual["total_pairs"], expected_total)
        self.assertEqual(gradual["fixed_pairs"], expected_fixed)
        self.assertEqual(gradual["broken_pairs"], expected_broken)
        self.assertGreater(expected_total, 0)

    def test_pair_diagnostics_ignore_c0_ties_and_c6_ties_antigravity(self) -> None:
        """Break caught: tie pairs must enter neither fixed nor broken bucket."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
            _pair_diagnostics_for_rows_antigravity,
            Exp464MethodFoldMetricAntigravity,
        )

        def row_antigravity(
            *,
            method: str,
            normal_scores: tuple[float, ...],
            gradual_scores: tuple[float, ...],
        ) -> Exp464MethodFoldMetricAntigravity:
            primary_scores = {
                **_primary_scores_antigravity(),
                "gradual_drift": gradual_scores,
            }
            evidence = (
                _c6_row_evidence_antigravity(
                    normal_scores,
                    primary_scores,
                    normal_confirmed=2,
                    primary_confirmed=2,
                )
                if method == "C6"
                else {
                    "fit_amplitudes": (),
                    "query_amplitudes": {},
                    "query_cosines": {},
                    "raw_evidence_kind": "",
                    "raw_evidence_source_method": "",
                    "raw_fit_vector_blob": "",
                    "raw_query_vector_blobs": {},
                }
            )
            return Exp464MethodFoldMetricAntigravity(
                method=method,
                seed=20260721,
                fold_index=0,
                normal_trials=len(normal_scores),
                normal_alerts=0,
                normal_wilson_upper=0.0,
                per_type_auc={name: 1.0 for name in TYPES_ANTIGRAVITY},
                mean_auc=1.0,
                scores_finite=True,
                normal_scores=normal_scores,
                primary_scores=primary_scores,
                confirmation_counts={name: 0 for name in CONFIRMATION_TYPES_ANTIGRAVITY},
                confirmation_rates={name: 0.0 for name in CONFIRMATION_TYPES_ANTIGRAVITY},
                pair_diagnostics={
                    change_type: {"total_pairs": 0, "fixed_pairs": 0, "broken_pairs": 0}
                    for change_type in PAIR_TYPES_ANTIGRAVITY
                },
                passed=True,
                failure_reasons=(),
                feature_dimension=1,
                provenance="pair_fixture",
                fit_amplitudes=evidence["fit_amplitudes"],
                query_amplitudes=evidence["query_amplitudes"],
                query_cosines=evidence["query_cosines"],
                raw_evidence_kind=evidence["raw_evidence_kind"],
                raw_evidence_source_method=evidence["raw_evidence_source_method"],
                raw_fit_vector_blob=evidence["raw_fit_vector_blob"],
                raw_query_vector_blobs=evidence["raw_query_vector_blobs"],
            )

        c0_tie = _pair_diagnostics_for_rows_antigravity(
            row_antigravity(
                method="C0",
                normal_scores=(0.1, 0.1),
                gradual_scores=(0.1, 0.1),
            ),
            row_antigravity(
                method="C6",
                normal_scores=(0.2, 0.3),
                gradual_scores=(0.8, 0.9),
            ),
        )["gradual_drift"]
        self.assertEqual(c0_tie["fixed_pairs"], 0)
        self.assertEqual(c0_tie["broken_pairs"], 0)

        c6_tie = _pair_diagnostics_for_rows_antigravity(
            row_antigravity(
                method="C0",
                normal_scores=(0.1, 0.2),
                gradual_scores=(0.8, 0.9),
            ),
            row_antigravity(
                method="C6",
                normal_scores=(0.2, 0.2),
                gradual_scores=(0.2, 0.2),
            ),
        )["gradual_drift"]
        self.assertEqual(c6_tie["fixed_pairs"], 0)
        self.assertEqual(c6_tie["broken_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
