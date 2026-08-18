from __future__ import annotations

import unittest

from experiments.exp459_ecg_normal2_blocked_validation_antigravity.tests.test_artifacts_antigravity import (
    decision_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_exports_only_aggregate_train_payload_and_exp458_context_antigravity(self) -> None:
        """Break caught: dashboard leaks raw rows, labels, DB paths, or omits fixed predecessor metrics."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        dashboard = export_dashboard_antigravity(decision_antigravity(unstable=True))
        self.assertEqual(
            set(dashboard),
            {
                "experiment_id",
                "progress",
                "status",
                "passed",
                "label_query_count",
                "fold_records",
                "seed_summaries",
                "exp458_predecessor",
            },
        )
        self.assertEqual(dashboard["experiment_id"], "Exp459")
        self.assertEqual(dashboard["progress"], "Task 4/6")
        self.assertEqual(dashboard["status"], "unstable")
        self.assertFalse(dashboard["passed"])
        self.assertEqual(dashboard["label_query_count"], 0)
        self.assertEqual(len(dashboard["fold_records"]), 10)
        self.assertEqual(len(dashboard["seed_summaries"]), 2)
        self.assertEqual(
            tuple(tuple(row) for row in dashboard["fold_records"]),
            (("dataset", "seed", "fold_index", "normal_trials", "normal_alerts", "normal_wilson_upper", "per_type_auc", "pooled_auc", "passed"),) * 10,
        )
        self.assertEqual(
            tuple(tuple(row) for row in dashboard["seed_summaries"]),
            (("dataset", "seed", "fold_count", "passed"), ("dataset", "seed", "fold_count", "passed")),
        )
        self.assertEqual(
            dashboard["exp458_predecessor"],
            {"experiment": "Exp458", "f1": 0.5796148426491311, "fn": 829, "fp": 66, "tn": 70788, "tp": 617},
        )

        rendered = repr(dashboard)
        for forbidden in (
            "fit_indices",
            "query_indices",
            "values_blob",
            "raw",
            "test_labels",
            "TEST",
            "/Users/mac/Documents/Dataset",
            "univariate_ts.db",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered)

    def test_dashboard_rejects_forged_result_and_mutated_nested_fields_antigravity(self) -> None:
        """Break caught: dashboard accepts caller-minted decisions or nonvalidated nested values."""
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.contracts_antigravity import (
            ValidationDecisionAntigravity,
        )
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        decision = decision_antigravity()
        forged = ValidationDecisionAntigravity(
            dataset=decision.dataset,
            qualification=decision.qualification,
            status=decision.status,
            passed=decision.passed,
        )
        with self.assertRaises(ValueError):
            export_dashboard_antigravity(forged)

        mutated = decision_antigravity()
        object.__setattr__(mutated, "status", "unstable")
        object.__setattr__(mutated, "passed", False)
        with self.assertRaises(ValueError):
            export_dashboard_antigravity(mutated)


if __name__ == "__main__":
    unittest.main()
