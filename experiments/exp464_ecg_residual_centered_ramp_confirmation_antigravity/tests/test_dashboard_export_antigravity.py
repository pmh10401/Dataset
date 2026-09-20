from __future__ import annotations

import json
import unittest

from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.tests.test_qualification_antigravity import (
    qualification_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_is_aggregate_only_and_scalar_safe_antigravity(self) -> None:
        """Break caught: dashboard leaks raw scores, paths, or caller strings."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(qualification, authority)
        dashboard = export_dashboard_antigravity(qualification, decision)
        self.assertEqual(
            tuple(dashboard),
            (
                "primary_candidate",
                "primary_passed_count",
                "primary_required_count",
                "eligible_for_routing_research",
                "method_summaries",
                "failed_fold_count",
                "historical_period_floors",
                "non_degradation_counts",
                "label_query_count",
            ),
        )
        self.assertEqual(dashboard["primary_candidate"], "template_residual_centered_ramp_confirmed_max")
        self.assertIsInstance(dashboard["primary_passed_count"], int)
        self.assertIsInstance(dashboard["primary_required_count"], int)
        self.assertIsInstance(dashboard["eligible_for_routing_research"], bool)
        self.assertEqual(len(dashboard["method_summaries"]), 9)
        self.assertEqual(
            tuple(dashboard["method_summaries"][0]),
            (
                "method",
                "fold_count",
                "passed_count",
                "failed_count",
                "min_level_shift_auc",
                "min_gradual_drift_auc",
                "min_amplitude_scaling_auc",
                "min_period_change_auc",
                "min_mean_auc",
                "max_normal_wilson_upper",
                "all_scores_finite",
                "passed",
            ),
        )
        self.assertEqual(
            tuple(dashboard["historical_period_floors"][0]),
            (
                "seed",
                "fold_index",
                "required_period_change_auc_min",
                "candidate_period_change_auc",
            ),
        )
        self.assertEqual(
            tuple(dashboard["non_degradation_counts"]),
            ("gradual_rows_not_below_c0", "period_rows_not_below_c0"),
        )
        payload = json.dumps(dashboard, ensure_ascii=False, sort_keys=True)
        for forbidden in (
            "normal_scores",
            "primary_scores",
            "fit_indices",
            "query_indices",
            "feature_dimension",
            "provenance",
            "/Users/mac",
            "label_sql",
            "ECG5000_normal_2",
            "Task4 implemented pending fresh review",
            "qualification_failed_blocked_before_routing_research",
            "result_scope",
            "stage",
            "status",
        ):
            self.assertNotIn(forbidden, payload)

    def test_dashboard_uses_only_registered_keys_and_scalar_rows_antigravity(self) -> None:
        """Break caught: dashboard invents free-form fields or exposes non-scalar payload values."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )

        authority = capture_exp463_authority_antigravity()
        qualification = qualification_antigravity()
        decision = evaluate_qualification_antigravity(qualification, authority)
        dashboard = export_dashboard_antigravity(qualification, decision)
        self.assertEqual(
            set(dashboard),
            {
                "primary_candidate",
                "primary_passed_count",
                "primary_required_count",
                "eligible_for_routing_research",
                "method_summaries",
                "failed_fold_count",
                "historical_period_floors",
                "non_degradation_counts",
                "label_query_count",
            },
        )
        for key in (
            "primary_passed_count",
            "primary_required_count",
            "failed_fold_count",
            "label_query_count",
        ):
            self.assertIsInstance(dashboard[key], int)
        self.assertIsInstance(dashboard["primary_candidate"], str)
        self.assertIsInstance(dashboard["eligible_for_routing_research"], bool)
        for summary in dashboard["method_summaries"]:
            self.assertEqual(
                set(summary),
                {
                    "method",
                    "fold_count",
                    "passed_count",
                    "failed_count",
                    "min_level_shift_auc",
                    "min_gradual_drift_auc",
                    "min_amplitude_scaling_auc",
                    "min_period_change_auc",
                    "min_mean_auc",
                    "max_normal_wilson_upper",
                    "all_scores_finite",
                    "passed",
                },
            )
            self.assertEqual(
                {type(summary[key]) for key in summary},
                {str, int, float, bool},
            )
        for floor in dashboard["historical_period_floors"]:
            self.assertEqual(
                set(floor),
                {
                    "seed",
                    "fold_index",
                    "required_period_change_auc_min",
                    "candidate_period_change_auc",
                },
            )
            self.assertEqual(type(floor["seed"]), int)
            self.assertEqual(type(floor["fold_index"]), int)
            self.assertEqual(type(floor["required_period_change_auc_min"]), float)
            self.assertEqual(type(floor["candidate_period_change_auc"]), float)
        self.assertEqual(
            set(dashboard["non_degradation_counts"]),
            {"gradual_rows_not_below_c0", "period_rows_not_below_c0"},
        )
        self.assertTrue(
            all(
                type(value) is int
                for value in dashboard["non_degradation_counts"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
