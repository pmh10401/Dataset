from __future__ import annotations

from dataclasses import replace
from math import sqrt
import unittest

import numpy as np


METHODS_ANTIGRAVITY = ("M0", "M1", "M2", "M3", "M4", "M5", "M6")
TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def train_antigravity() -> np.ndarray:
    values = np.linspace(0.0, 1.0, 1413 * 140, dtype=np.float32).reshape(1413, 140)
    values.setflags(write=False)
    return values


def fold_scorer_antigravity(
    _train: np.ndarray,
    _fit_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    seed: int,
    fold_index: int,
):
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.scoring_antigravity import (
        FoldMethodScoresAntigravity,
    )

    del seed, fold_index
    size = len(query_indices)
    normal = np.linspace(0.0, 1.0, size, dtype=np.float64)
    primary = {
        change_type: normal + 10.0 + index
        for index, change_type in enumerate(TYPES_ANTIGRAVITY)
    }
    return FoldMethodScoresAntigravity(
        methods=METHODS_ANTIGRAVITY,
        original={method: normal for method in METHODS_ANTIGRAVITY},
        primary={method: primary for method in METHODS_ANTIGRAVITY},
        feature_dimensions={method: 1 for method in METHODS_ANTIGRAVITY},
        provenance={method: "synthetic_task4_score_evidence" for method in METHODS_ANTIGRAVITY},
    )


def qualification_antigravity():
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
        capture_exp462_authority_antigravity,
    )
    from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
        _qualify_methods_with_dependencies_antigravity,
    )

    return _qualify_methods_with_dependencies_antigravity(
        authority=capture_exp462_authority_antigravity(),
        loader=train_antigravity,
        fold_scorer=fold_scorer_antigravity,
    )


class QualificationAntigravityTest(unittest.TestCase):
    def test_qualification_emits_exact_ordered_70_rows_and_seven_summaries_antigravity(self) -> None:
        """Break caught: a method, seed, fold, or summary is silently dropped."""
        result = qualification_antigravity()
        self.assertEqual(result.method_order, METHODS_ANTIGRAVITY)
        self.assertEqual(len(result.fold_metrics), 70)
        self.assertEqual(
            tuple((row.method, row.seed, row.fold_index) for row in result.fold_metrics),
            tuple(
                (method, seed, fold)
                for method in METHODS_ANTIGRAVITY
                for seed in (20260721, 20261721)
                for fold in range(5)
            ),
        )
        self.assertEqual(tuple(summary.method for summary in result.summaries), METHODS_ANTIGRAVITY)
        for row in result.fold_metrics:
            self.assertEqual(tuple(row.per_type_auc), TYPES_ANTIGRAVITY)
            self.assertEqual(tuple(row.primary_scores), TYPES_ANTIGRAVITY)
            self.assertTrue(row.scores_finite)

    def test_normalization_recomputes_metrics_and_ignores_caller_pass_boolean_antigravity(self) -> None:
        """Break caught: stored AUCs or pass flags can forge the M2 decision."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        qualification = qualification_antigravity()
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "M2" and row.seed == 20260721 and row.fold_index == 3
        )
        row = rows[index]
        forged = replace(
            row,
            passed=True,
            failure_reasons=(),
            primary_scores={
                **row.primary_scores,
                "period_change": row.normal_scores,
            },
        )
        rows[index] = forged
        altered = replace(qualification, fold_metrics=tuple(rows))
        decision = evaluate_qualification_antigravity(
            capture_exp462_authority_antigravity(), altered
        )
        self.assertFalse(decision.eligible_for_routing_research)
        self.assertIn("primary_candidate_not_10_of_10", decision.failure_reasons)
        self.assertIn("period_change_auc_below_global_min", decision.failure_reasons)
        self.assertIn("historical_period_change_below_min:20260721:3", decision.failure_reasons)

    def test_m0_m6_control_reproduction_is_required_antigravity(self) -> None:
        """Break caught: Exp463 silently accepts drifted predecessor controls."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
            validate_control_reproduction_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )

        self.assertEqual(
            validate_control_reproduction_antigravity(
                qualification_antigravity(), capture_exp462_authority_antigravity()
            ),
            ("m0_control_not_reproduced", "m6_control_not_reproduced"),
        )

    def test_metric_recomputation_has_independent_auc_pooled_wilson_and_finite_checks_antigravity(self) -> None:
        """Break caught: stored metric scalars or finite flags replace score evidence."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.contracts_antigravity import (
            Exp463MethodFoldMetricAntigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
            recompute_metric_from_scores_antigravity,
        )

        authority = capture_exp462_authority_antigravity()
        normal = (0.0, 1.0)
        primary = {
            "level_shift": (1.0, 2.0),
            "gradual_drift": (0.0, 2.0),
            "amplitude_scaling": (0.0, 1.0),
            "period_change": (2.0, 3.0),
        }
        row = Exp463MethodFoldMetricAntigravity(
            method="M1",
            seed=20260721,
            fold_index=0,
            normal_trials=2,
            normal_alerts=0,
            normal_wilson_upper=0.0,
            per_type_auc={name: 0.5 for name in TYPES_ANTIGRAVITY},
            pooled_auc=0.5,
            scores_finite=True,
            normal_scores=normal,
            primary_scores=primary,
            passed=False,
            failure_reasons=("pending",),
        )
        recomputed = recompute_metric_from_scores_antigravity(row, authority)

        expected_auc: dict[str, float] = {}
        normal_array = np.asarray(normal, dtype=np.float64)
        for name in TYPES_ANTIGRAVITY:
            anomaly = np.asarray(primary[name], dtype=np.float64)
            wins = sum(int(anomaly_value > normal_value) for anomaly_value in anomaly for normal_value in normal_array)
            ties = sum(int(anomaly_value == normal_value) for anomaly_value in anomaly for normal_value in normal_array)
            expected_auc[name] = (wins + 0.5 * ties) / (len(anomaly) * len(normal_array))
        expected_pooled = sum(expected_auc.values()) / len(expected_auc)
        ordered = np.sort(normal_array)
        counts = len(ordered) - np.searchsorted(ordered, normal_array, side="left")
        expected_alerts = int(
            np.count_nonzero((1.0 + counts) / (1.0 + len(ordered)) <= 0.005)
        )
        z = 1.959963984540054
        proportion = expected_alerts / len(ordered)
        z_squared = z**2
        denominator = 1.0 + z_squared / len(ordered)
        center = proportion + z_squared / (2.0 * len(ordered))
        adjustment = z * sqrt(
            (proportion * (1.0 - proportion) + z_squared / (4.0 * len(ordered)))
            / len(ordered)
        )
        expected_wilson = min(1.0, (center + adjustment) / denominator)

        self.assertEqual(dict(recomputed.per_type_auc), expected_auc)
        self.assertEqual(recomputed.pooled_auc, expected_pooled)
        self.assertEqual(recomputed.normal_alerts, expected_alerts)
        self.assertAlmostEqual(recomputed.normal_wilson_upper, expected_wilson, places=15)
        self.assertTrue(recomputed.scores_finite)

        nonfinite = replace(
            row,
            normal_scores=(float("nan"), 1.0),
            primary_scores={name: (float("nan"), 1.0) for name in TYPES_ANTIGRAVITY},
        )
        nonfinite_recomputed = recompute_metric_from_scores_antigravity(nonfinite, authority)
        self.assertFalse(nonfinite_recomputed.scores_finite)
        self.assertEqual(dict(nonfinite_recomputed.per_type_auc), {name: 0.0 for name in TYPES_ANTIGRAVITY})

    def test_m2_gates_use_exp463_boundaries_and_m6_keeps_predecessor_boundaries_antigravity(self) -> None:
        """Break caught: Exp462 historical thresholds leak into M2 qualification."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.predecessor_antigravity import (
            capture_exp462_authority_antigravity,
        )
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.qualification_antigravity import (
            recompute_row_pass_antigravity,
        )

        authority = capture_exp462_authority_antigravity()
        qualification = qualification_antigravity()
        config = FROZEN_CONFIG_ANTIGRAVITY
        base = next(row for row in qualification.fold_metrics if row.method == "M2")

        passed, reasons = recompute_row_pass_antigravity(base, authority)
        self.assertTrue(passed)
        self.assertEqual(reasons, ())
        primary_summary = next(summary for summary in qualification.summaries if summary.method == "M2")
        self.assertEqual(primary_summary.fold_count, 10)
        self.assertEqual(primary_summary.passed_count, 10)
        self.assertEqual(primary_summary.failed_count, 0)
        self.assertTrue(primary_summary.all_scores_finite)

        for change_type in TYPES_ANTIGRAVITY:
            metrics = dict(base.per_type_auc)
            metrics[change_type] = config.per_type_auc_min - 1e-9
            candidate = replace(
                base,
                per_type_auc=metrics,
                pooled_auc=float(sum(metrics.values()) / len(metrics)),
            )
            passed, reasons = recompute_row_pass_antigravity(candidate, authority)
            self.assertFalse(passed)
            self.assertIn(f"{change_type}_auc_below_min", reasons)

        pooled_metrics = {name: config.pooled_auc_min - 1e-9 for name in TYPES_ANTIGRAVITY}
        pooled_candidate = replace(
            base,
            per_type_auc=pooled_metrics,
            pooled_auc=float(sum(pooled_metrics.values()) / len(pooled_metrics)),
        )
        passed, reasons = recompute_row_pass_antigravity(pooled_candidate, authority)
        self.assertFalse(passed)
        self.assertIn("pooled_auc_below_min", reasons)

        wilson_candidate = replace(
            base,
            normal_wilson_upper=config.normal_wilson_upper_max + 1e-9,
        )
        passed, reasons = recompute_row_pass_antigravity(wilson_candidate, authority)
        self.assertFalse(passed)
        self.assertIn("normal_wilson_upper_above_max", reasons)

        finite_candidate = replace(base, scores_finite=False)
        passed, reasons = recompute_row_pass_antigravity(finite_candidate, authority)
        self.assertFalse(passed)
        self.assertIn("nonfinite_scores", reasons)

        gradual_candidate = replace(
            base,
            per_type_auc={**base.per_type_auc, "gradual_drift": config.gradual_drift_auc_strictly_above_exp462},
            pooled_auc=float(
                (sum(base.per_type_auc.values()) - base.per_type_auc["gradual_drift"] + config.gradual_drift_auc_strictly_above_exp462)
                / len(base.per_type_auc)
            ),
        )
        passed, reasons = recompute_row_pass_antigravity(gradual_candidate, authority)
        self.assertFalse(passed)
        self.assertIn("gradual_drift_auc_below_min", reasons)
        self.assertIn("gradual_drift_not_strictly_improved_over_exp462", reasons)

        period_candidate = replace(
            base,
            per_type_auc={**base.per_type_auc, "period_change": config.period_change_auc_global_min - 1e-9},
            pooled_auc=float(
                (sum(base.per_type_auc.values()) - base.per_type_auc["period_change"] + config.period_change_auc_global_min - 1e-9)
                / len(base.per_type_auc)
            ),
        )
        passed, reasons = recompute_row_pass_antigravity(period_candidate, authority)
        self.assertFalse(passed)
        self.assertIn("period_change_auc_below_global_min", reasons)

        for historical in config.historical_period_fold_thresholds:
            historical_base = next(
                row
                for row in qualification.fold_metrics
                if row.method == "M2"
                and row.seed == historical.seed
                and row.fold_index == historical.fold_index
            )
            equal_metrics = {
                **historical_base.per_type_auc,
                "period_change": historical.period_change_auc_min,
            }
            equal_row = replace(
                historical_base,
                per_type_auc=equal_metrics,
                pooled_auc=float(sum(equal_metrics.values()) / len(equal_metrics)),
            )
            passed, reasons = recompute_row_pass_antigravity(equal_row, authority)
            self.assertTrue(passed, (historical.seed, historical.fold_index, reasons))
            below_metrics = {
                **historical_base.per_type_auc,
                "period_change": historical.period_change_auc_min - 1e-9,
            }
            below_row = replace(
                historical_base,
                per_type_auc=below_metrics,
                pooled_auc=float(sum(below_metrics.values()) / len(below_metrics)),
            )
            passed, reasons = recompute_row_pass_antigravity(below_row, authority)
            self.assertFalse(passed)
            self.assertIn(
                f"historical_period_change_below_min:{historical.seed}:{historical.fold_index}",
                reasons,
            )

        # The second M2 threshold is intentionally below the predecessor's
        # 0.9285624465570143 value; this midpoint must still pass M2.
        second = config.historical_period_fold_thresholds[1]
        predecessor_second = authority.historical_period_folds[1]
        midpoint = (second.period_change_auc_min + predecessor_second.period_change_auc) / 2.0
        midpoint_base = next(
            row
            for row in qualification.fold_metrics
            if row.method == "M2" and row.seed == second.seed and row.fold_index == second.fold_index
        )
        midpoint_metrics = {**midpoint_base.per_type_auc, "period_change": midpoint}
        midpoint_row = replace(
            midpoint_base,
            per_type_auc=midpoint_metrics,
            pooled_auc=float(sum(midpoint_metrics.values()) / len(midpoint_metrics)),
        )
        passed, reasons = recompute_row_pass_antigravity(midpoint_row, authority)
        self.assertTrue(passed, reasons)

        m6_base = next(
            row
            for row in qualification.fold_metrics
            if row.method == "M6" and row.seed == predecessor_second.seed and row.fold_index == predecessor_second.fold_index
        )
        m6_equal_metrics = {**m6_base.per_type_auc, "period_change": predecessor_second.period_change_auc}
        m6_equal = replace(
            m6_base,
            per_type_auc=m6_equal_metrics,
            pooled_auc=float(sum(m6_equal_metrics.values()) / len(m6_equal_metrics)),
        )
        passed, reasons = recompute_row_pass_antigravity(m6_equal, authority)
        self.assertTrue(passed, reasons)
        m6_exp463_metrics = {**m6_base.per_type_auc, "period_change": second.period_change_auc_min}
        m6_exp463 = replace(
            m6_base,
            per_type_auc=m6_exp463_metrics,
            pooled_auc=float(sum(m6_exp463_metrics.values()) / len(m6_exp463_metrics)),
        )
        passed, reasons = recompute_row_pass_antigravity(m6_exp463, authority)
        self.assertFalse(passed)
        self.assertIn(
            f"m6_historical_period_change_below_min:{predecessor_second.seed}:{predecessor_second.fold_index}",
            reasons,
        )


if __name__ == "__main__":
    unittest.main()
