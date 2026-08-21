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


if __name__ == "__main__":
    unittest.main()
