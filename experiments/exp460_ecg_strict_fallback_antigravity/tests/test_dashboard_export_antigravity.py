from __future__ import annotations

import unittest

from experiments.exp460_ecg_strict_fallback_antigravity.tests.test_artifacts_antigravity import (
    authority_antigravity,
    result_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_exports_only_allowlisted_scientific_payload_antigravity(self) -> None:
        """Break caught: dashboard leaks raw rows, paths, labels, or omits the fixed Task 4 summary fields."""
        from experiments.exp460_ecg_strict_fallback_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        dashboard = export_dashboard_antigravity(authority_antigravity(), result_antigravity())
        self.assertEqual(
            set(dashboard),
            {
                "comparison_rows",
                "dataset",
                "experiment_id",
                "failed_folds",
                "implementation_official",
                "joint_success",
                "label_query_count",
                "official_evaluate_invocations",
                "official_inspect_invocations",
                "predecessor_authority_digest",
                "predecessor_experiment_id",
                "progress",
                "status",
                "strict_route",
                "strict_status",
                "worst_period_change_auc",
            },
        )
        self.assertEqual(dashboard["experiment_id"], "Exp460")
        self.assertEqual(dashboard["dataset"], "ECG5000_normal_2")
        self.assertEqual(dashboard["predecessor_experiment_id"], "Exp459")
        self.assertEqual(
            dashboard["predecessor_authority_digest"],
            "f661c4ce6c7fd6625ac5cf767db688ce7ede7a0d0557721c9e9fa7e7ea6508ee",
        )
        self.assertEqual(dashboard["progress"], "Task 4/6")
        self.assertEqual(dashboard["status"], "fallback_triggered_performance_not_improved")
        self.assertFalse(dashboard["implementation_official"])
        self.assertFalse(dashboard["joint_success"])
        self.assertEqual(dashboard["label_query_count"], 0)
        self.assertEqual(dashboard["official_evaluate_invocations"], 0)
        self.assertEqual(dashboard["official_inspect_invocations"], 0)
        self.assertEqual(dashboard["strict_route"], "baseline")
        self.assertEqual(
            dashboard["strict_status"],
            "fallback_triggered_performance_not_improved",
        )
        self.assertEqual(dashboard["worst_period_change_auc"], 0.8004124541019063)
        self.assertEqual(len(dashboard["comparison_rows"]), 4)
        self.assertEqual(len(dashboard["failed_folds"]), 2)
        self.assertEqual(
            dashboard["comparison_rows"][0],
            {
                "f1": 0.5742574257425742,
                "fn": 837,
                "fp": 66,
                "route": "baseline",
                "rule": "always_baseline",
                "tn": 70788,
                "tp": 609,
            },
        )

        rendered = repr(dashboard)
        for forbidden in (
            "TEST",
            "labels",
            "fold_records",
            "seed_summaries",
            "fit_indices",
            "query_indices",
            "/Users/mac/Documents",
            "runtime_executable",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered)

    def test_dashboard_rejects_inconsistent_result_and_nested_mutation_antigravity(self) -> None:
        """Break caught: dashboard accepts semantically inconsistent result objects."""
        from experiments.exp460_ecg_strict_fallback_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        authority = authority_antigravity()
        mutated = result_antigravity()
        object.__setattr__(mutated, "strict_route", "candidate")
        object.__setattr__(mutated, "joint_success", True)
        with self.assertRaises(ValueError):
            export_dashboard_antigravity(authority, mutated)


if __name__ == "__main__":
    unittest.main()
