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
                "experiment",
                "dataset",
                "stage",
                "primary_candidate",
                "primary_status",
                "method_order",
                "method_summaries",
                "failed_fold_count",
                "historical_period_floors",
                "non_degradation",
                "decision",
                "label_query_count",
                "result_scope",
            ),
        )
        self.assertEqual(tuple(dashboard["method_order"]), ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"))
        self.assertEqual(len(dashboard["method_summaries"]), 9)
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
        ):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
