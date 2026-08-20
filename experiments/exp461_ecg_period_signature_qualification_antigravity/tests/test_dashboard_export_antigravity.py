from __future__ import annotations

import json
import unittest

from experiments.exp461_ecg_period_signature_qualification_antigravity.tests.test_artifacts_antigravity import (
    publication_inputs_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_payload_is_aggregate_only_and_label_free_antigravity(self) -> None:
        """Break caught: dashboard exposes raw series, dataset indices, paths, TEST labels, or secrets."""
        from experiments.exp461_ecg_period_signature_qualification_antigravity.dashboard_export_antigravity import (
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
        self.assertEqual(dashboard["failed_fold_count"], 3)
        self.assertEqual(dashboard["decision"]["eligible_for_routing_research"], True)
        payload = json.dumps(dashboard, ensure_ascii=False, sort_keys=True)
        for forbidden in (
            "/Users/mac",
            "fit_indices",
            "query_indices",
            "raw",
            "series",
            "TEST",
            "label_sql",
            "secret",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, payload)


if __name__ == "__main__":
    unittest.main()
