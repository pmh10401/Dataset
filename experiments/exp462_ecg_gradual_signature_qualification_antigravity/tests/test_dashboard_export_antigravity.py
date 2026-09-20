from __future__ import annotations

import json
import unittest

from experiments.exp462_ecg_gradual_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
    publication_inputs_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_is_aggregate_only_and_exposes_four_type_minima_antigravity(self) -> None:
        """Break caught: dashboard exposes raw evidence, paths, labels, or secrets."""
        from experiments.exp462_ecg_gradual_signature_qualification_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        authority, qualification, decision = publication_inputs_antigravity()
        dashboard = export_dashboard_antigravity(authority, qualification, decision)
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
                "historical_improvements",
                "decision",
                "label_query_count",
                "official_command_count",
                "result_scope",
            ),
        )
        self.assertEqual(dashboard["stage"], "Task4 implemented pending external review / Task5 next")
        self.assertEqual(dashboard["label_query_count"], 0)
        self.assertEqual(dashboard["official_command_count"], 0)
        self.assertEqual(dashboard["failed_fold_count"], 0)
        self.assertTrue(dashboard["decision"]["eligible_for_routing_research"])
        self.assertEqual(
            tuple(dashboard["method_summaries"][0]["min_per_type_auc"]),
            ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
        )
        payload = json.dumps(dashboard, ensure_ascii=False, sort_keys=True)
        for forbidden in ("/Users/mac", "normal_scores", "primary_scores", "fit_indices", "query_indices", "TEST", "label_sql"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
