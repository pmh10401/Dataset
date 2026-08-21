from __future__ import annotations

from dataclasses import replace
from math import sqrt
import unittest

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
        feature_dimensions={method: int(method_payloads[method].get("feature_dimension", 1)) for method in METHODS_ANTIGRAVITY},
        provenance={method: str(method_payloads[method].get("provenance", f"fixture_{method.lower()}")) for method in METHODS_ANTIGRAVITY},
    )


def synthetic_fold_payload_antigravity(
    seed: int,
    fold_index: int,
) -> dict[str, dict[str, object]]:
    del seed
    c0_normal = (0.10, 0.20)
    c0_primary = _primary_scores_antigravity()
    c6_gradual = (0.94, 0.98)
    c6_period = (0.95, 0.99)
    if fold_index == 3:
        c6_period = (0.907392736783864, 0.99)
    if fold_index == 4:
        c6_period = (0.9216148584075248, 0.99)
    c6_primary = _primary_scores_antigravity(
        gradual=c6_gradual,
        period=c6_period,
    )
    controls = {
        "C0": {
            "normal": c0_normal,
            "primary": c0_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "exp463_m0_control",
        },
        "C1": {
            "normal": c0_normal,
            "primary": c0_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "exp463_m2_control",
        },
        "C2": {
            "normal": c0_normal,
            "primary": c0_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "exp463_m4_control",
        },
        "C7": {
            "normal": c0_normal,
            "primary": c0_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "exp463_m6_control",
        },
    }
    rows = {
        "C0": controls["C0"],
        "C1": controls["C1"],
        "C2": controls["C2"],
        "C3": {
            "normal": c0_normal,
            "primary": _primary_scores_antigravity(gradual=(0.88, 0.94)),
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "rank_only",
        },
        "C4": {
            "normal": c0_normal,
            "primary": _primary_scores_antigravity(gradual=(0.89, 0.95)),
            "confirmation_counts": _confidence_counts_antigravity(normal=0, gradual=0, period=0, level=0, amplitude=0),
            "provenance": "always_max",
        },
        "C5": {
            "normal": c0_normal,
            "primary": _primary_scores_antigravity(period=(0.91, 0.97)),
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "raw_centered",
        },
        "C6": {
            "normal": c0_normal,
            "primary": c6_primary,
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "template_residual_centered",
        },
        "C7": controls["C7"],
        "C8": {
            "normal": c0_normal,
            "primary": _primary_scores_antigravity(period=(0.90, 0.96)),
            "confirmation_counts": _confidence_counts_antigravity(normal=1, gradual=2, period=2, level=2, amplitude=2),
            "feature_dimension": 2,
            "provenance": "uncentered",
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
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
        capture_exp463_authority_antigravity,
    )
    from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp463_authority_antigravity(),
        loader=train_antigravity,
        fold_scorer=lambda train, fit_indices, query_indices, *, seed, fold_index: fold_scorer_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=seed,
            fold_index=fold_index,
            mutated_payloads=mutated_payloads,
        ),
    )


class QualificationAntigravityTest(unittest.TestCase):
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
            failure_reasons=("forged",),
            feature_dimension=2,
            provenance="fixture",
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

    def test_normalization_rejects_wrong_order_identity_and_type_and_recomputes_c6_against_c0_antigravity(self) -> None:
        """Break caught: reordered rows, wrong IDs, or forged booleans survive normalization."""
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
        normalized = normalize_qualification_antigravity(
            replace(qualification, fold_metrics=tuple(rows)),
            authority,
        )
        c6_row = next(
            row
            for row in normalized.fold_metrics
            if row.method == "C6" and row.seed == 20260721 and row.fold_index == 3
        )
        self.assertFalse(c6_row.passed)
        self.assertIn("period_change_auc_below_c0", c6_row.failure_reasons)
        self.assertIn("historical_period_change_below_min:20260721:3", c6_row.failure_reasons)

        reordered = list(qualification.fold_metrics)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaises(ValueError):
            normalize_qualification_antigravity(
                replace(qualification, fold_metrics=tuple(reordered)),
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
        self.assertEqual(expected_total, 4)
        self.assertEqual(expected_fixed, 0)
        self.assertEqual(expected_broken, 0)


if __name__ == "__main__":
    unittest.main()
