from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity import (
    TYPES_ANTIGRAVITY,
    qualification_antigravity,
)


class EvaluationAntigravityTest(unittest.TestCase):
    def test_decision_is_fixed_c6_only_and_blocks_before_routing_research_antigravity(self) -> None:
        """Break caught: Task 4 promotes another method or opens routing before review."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(qualification, authority)
        self.assertEqual(decision.primary_candidate, "template_residual_centered_ramp_confirmed_max")
        self.assertEqual(decision.primary_required_count, 10)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(decision.failure_reasons, ())
        self.assertTrue(decision.eligible_for_routing_research)
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)
        self.assertEqual(
            decision.status,
            "qualification_passed_ready_for_task5_reproducibility",
        )

    def test_decision_recomputes_c6_gates_from_score_evidence_antigravity(self) -> None:
        """Break caught: caller summaries or booleans bypass the frozen C6 gates."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        expected_decision = evaluate_qualification_antigravity(qualification, authority)
        rows = list(qualification.fold_metrics)
        index = next(
            i
            for i, row in enumerate(rows)
            if row.method == "C6" and row.seed == 20261721 and row.fold_index == 4
        )
        row = rows[index]
        rows[index] = replace(
            row,
            passed=True,
            failure_reasons=(),
        )
        decision = evaluate_qualification_antigravity(
            replace(qualification, fold_metrics=tuple(rows)),
            authority,
        )
        self.assertEqual(decision, expected_decision)

    def test_source_free_c6_gate_matrix_covers_every_frozen_decision_boundary_antigravity(self) -> None:
        """Break caught: a frozen C6 decision gate drifts, disappears, or re-enters TRAIN-backed qualification."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        baseline = qualification_antigravity()

        def c6_summary_antigravity(qualification):
            return next(summary for summary in qualification.summaries if summary.method == "C6")

        def replace_c6_summary_antigravity(qualification, **changes):
            summaries = list(qualification.summaries)
            index = next(
                i for i, summary in enumerate(summaries) if summary.method == "C6"
            )
            summaries[index] = replace(summaries[index], **changes)
            return replace(qualification, summaries=tuple(summaries))

        def replace_c6_row_antigravity(qualification, *, seed: int, fold_index: int, **changes):
            rows = list(qualification.fold_metrics)
            index = next(
                i
                for i, row in enumerate(rows)
                if row.method == "C6" and row.seed == seed and row.fold_index == fold_index
            )
            rows[index] = replace(rows[index], **changes)
            return replace(qualification, fold_metrics=tuple(rows))

        baseline_summary = c6_summary_antigravity(baseline)
        baseline_reasons = next(
            row.failure_reasons
            for row in baseline.fold_metrics
            if row.method == "C6" and row.seed == 20260721 and row.fold_index == 3
        )
        self.assertEqual(baseline_reasons, ())

        lowered_level_min = dict(baseline_summary.min_per_type_auc)
        lowered_level_min["level_shift"] = 0.849
        lowered_gradual_min = dict(baseline_summary.min_per_type_auc)
        lowered_gradual_min["gradual_drift"] = 0.849
        lowered_amplitude_min = dict(baseline_summary.min_per_type_auc)
        lowered_amplitude_min["amplitude_scaling"] = 0.849
        lowered_period_min = dict(baseline_summary.min_per_type_auc)
        lowered_period_min["period_change"] = 0.849

        cases = (
            ("pass", baseline, ()),
            (
                "primary_candidate_not_10_of_10",
                replace_c6_summary_antigravity(
                    baseline,
                    passed_count=9,
                    failed_count=1,
                    passed=False,
                ),
                ("primary_candidate_not_10_of_10",),
            ),
            (
                "primary_level_shift_auc_below_min",
                replace_c6_summary_antigravity(
                    baseline,
                    min_per_type_auc=lowered_level_min,
                ),
                ("primary_level_shift_auc_below_min",),
            ),
            (
                "primary_gradual_drift_auc_below_min",
                replace_c6_summary_antigravity(
                    baseline,
                    min_per_type_auc=lowered_gradual_min,
                ),
                ("primary_gradual_drift_auc_below_min",),
            ),
            (
                "primary_amplitude_scaling_auc_below_min",
                replace_c6_summary_antigravity(
                    baseline,
                    min_per_type_auc=lowered_amplitude_min,
                ),
                ("primary_amplitude_scaling_auc_below_min",),
            ),
            (
                "primary_period_change_auc_below_min",
                replace_c6_summary_antigravity(
                    baseline,
                    min_per_type_auc=lowered_period_min,
                ),
                ("primary_period_change_auc_below_min",),
            ),
            (
                "primary_mean_auc_below_0.90",
                replace_c6_summary_antigravity(baseline, min_mean_auc=0.899),
                ("primary_mean_auc_below_0.90",),
            ),
            (
                "primary_normal_wilson_upper_above_max",
                replace_c6_summary_antigravity(
                    baseline,
                    max_normal_wilson_upper=0.051,
                ),
                ("primary_normal_wilson_upper_above_max",),
            ),
            (
                "primary_scores_not_finite",
                replace_c6_summary_antigravity(baseline, all_scores_finite=False),
                ("primary_scores_not_finite",),
            ),
            (
                "primary_gradual_drift_min_below_0.85",
                replace_c6_summary_antigravity(
                    baseline,
                    min_gradual_drift_auc=0.849,
                ),
                ("primary_gradual_drift_min_below_0.85",),
            ),
            (
                "primary_gradual_drift_not_strictly_above_exp463_c0",
                replace_c6_summary_antigravity(
                    baseline,
                    min_gradual_drift_auc=0.8402306222021025,
                ),
                (
                    "primary_gradual_drift_min_below_0.85",
                    "primary_gradual_drift_not_strictly_above_exp463_c0",
                ),
            ),
            (
                "primary_gradual_drift_below_c0_on_some_fold",
                replace_c6_row_antigravity(
                    baseline,
                    seed=20260721,
                    fold_index=0,
                    failure_reasons=("gradual_drift_auc_below_c0",),
                ),
                ("primary_gradual_drift_below_c0_on_some_fold",),
            ),
            (
                "primary_period_change_min_below_exp463_global_floor",
                replace_c6_summary_antigravity(
                    baseline,
                    min_period_change_auc=0.9073927367838639,
                ),
                ("primary_period_change_min_below_exp463_global_floor",),
            ),
            (
                "primary_period_change_below_c0_on_some_fold",
                replace_c6_row_antigravity(
                    baseline,
                    seed=20261721,
                    fold_index=0,
                    failure_reasons=("period_change_auc_below_c0",),
                ),
                ("primary_period_change_below_c0_on_some_fold",),
            ),
            (
                "historical_period_change_below_min:20260721:3",
                replace_c6_row_antigravity(
                    baseline,
                    seed=20260721,
                    fold_index=3,
                    failure_reasons=("historical_period_change_below_min:20260721:3",),
                ),
                ("historical_period_change_below_min:20260721:3",),
            ),
            (
                "historical_period_change_below_min:20261721:4",
                replace_c6_row_antigravity(
                    baseline,
                    seed=20261721,
                    fold_index=4,
                    failure_reasons=("historical_period_change_below_min:20261721:4",),
                ),
                ("historical_period_change_below_min:20261721:4",),
            ),
        )

        for name, normalized, expected_reasons in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    tuple(c6_summary_antigravity(normalized).min_per_type_auc),
                    TYPES_ANTIGRAVITY,
                )
                with (
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity.normalize_qualification_antigravity",
                        return_value=normalized,
                    ),
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity.validate_control_reproduction_antigravity",
                        return_value=(),
                    ),
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.sources_antigravity.load_normal2_train_antigravity",
                        side_effect=AssertionError("source-free gate tests must not load TRAIN"),
                    ),
                    patch(
                        "experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity.score_fold_methods_antigravity",
                        side_effect=AssertionError("source-free gate tests must not call the production scorer"),
                    ),
                ):
                    decision = evaluate_qualification_antigravity(baseline, authority)
                self.assertEqual(decision.failure_reasons, expected_reasons)
                self.assertEqual(
                    decision.status,
                    (
                        "qualification_passed_ready_for_task5_reproducibility"
                        if not expected_reasons
                        else "qualification_failed_blocked_before_routing_research"
                    ),
                )
                self.assertEqual(
                    decision.eligible_for_routing_research,
                    not expected_reasons,
                )


if __name__ == "__main__":
    unittest.main()
