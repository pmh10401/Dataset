from __future__ import annotations

import json
import unittest

from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.tests.test_artifacts_antigravity import (
    publication_inputs_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_is_aggregate_only_and_exposes_ordered_four_type_minima_antigravity(self) -> None:
        """Break caught: dashboard leaks score evidence or drops an ordered method."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        dashboard = export_dashboard_antigravity(authority, qualification, decision)
        self.assertEqual(tuple(dashboard), (
            "experiment", "dataset", "stage", "primary_candidate", "primary_status",
            "method_order", "method_summaries", "failed_fold_count", "historical_improvements",
            "decision", "label_query_count", "official_command_count", "result_scope",
        ))
        self.assertEqual(tuple(dashboard["method_order"]), ("M0", "M1", "M2", "M3", "M4", "M5", "M6"))
        self.assertEqual(len(dashboard["method_summaries"]), 7)
        self.assertEqual(tuple(dashboard["method_summaries"][0]["min_per_type_auc"]), (
            "level_shift", "gradual_drift", "amplitude_scaling", "period_change",
        ))
        self.assertEqual(dashboard["label_query_count"], 0)
        self.assertEqual(dashboard["official_command_count"], 0)
        payload = json.dumps(dashboard, ensure_ascii=False, sort_keys=True)
        for forbidden in ("normal_scores", "primary_scores", "fit_indices", "query_indices", "TEST", "label_sql", "/Users/mac"):
            self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
