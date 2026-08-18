from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.tests.test_artifacts_antigravity import (
    publication_evidence_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_exports_only_aggregate_train_evidence_and_shared_validated_result_antigravity(self) -> None:
        """Break caught: dashboard exposes dataset rows or skips the shared semantic validator."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        result = publication_evidence_antigravity().result
        dashboard = export_dashboard_antigravity(result)
        self.assertEqual(set(dashboard), {"experiment_id", "status", "passed", "rank_evidence", "normal2_route", "methods", "criteria"})
        self.assertEqual(dashboard["experiment_id"], "Exp458")
        self.assertEqual(dashboard["normal2_route"]["dataset"], "ECG5000_normal_2")
        self.assertEqual(tuple(row["method"] for row in dashboard["methods"]), ("always_baseline", "family_wide_local_gap", "oracle_per_dataset", "normal2_rank_router"))
        self.assertEqual(tuple(row["metric"] for row in dashboard["criteria"]), ("final_fn", "final_fp", "final_f1"))
        self.assertEqual(len(dashboard["rank_evidence"]), 2)
        self.assertEqual(
            tuple(tuple(row) for row in dashboard["rank_evidence"]),
            (
                ("dataset", "seed", "normal_wilson_upper", "per_type_auc", "pooled_auc", "passed"),
                ("dataset", "seed", "normal_wilson_upper", "per_type_auc", "pooled_auc", "passed"),
            ),
        )
        self.assertEqual(
            tuple(tuple(row["per_type_auc"]) for row in dashboard["rank_evidence"]),
            (
                ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
                ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
            ),
        )
        self.assertEqual(
            tuple(
                (
                    type(row["dataset"]),
                    type(row["seed"]),
                    type(row["normal_wilson_upper"]),
                    type(row["per_type_auc"]),
                    type(row["pooled_auc"]),
                    type(row["passed"]),
                )
                for row in dashboard["rank_evidence"]
            ),
            ((str, int, float, dict, float, bool), (str, int, float, dict, float, bool)),
        )
        self.assertNotIn("normal_trials", repr(dashboard["rank_evidence"]))
        self.assertNotIn("normal_alerts", repr(dashboard["rank_evidence"]))
        self.assertNotIn("ECG5000_normal_1", repr(dashboard))
        self.assertNotIn("ChlorineConcentration_normal_1", repr(dashboard))

    def test_dashboard_rejects_forged_values_order_and_unapproved_dataset_names_antigravity(self) -> None:
        """Break caught: dashboard trusts a caller-mutated result or leaks row-level dataset identities."""
        from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        result = publication_evidence_antigravity().result
        with self.assertRaises(ValueError):
            export_dashboard_antigravity(replace(result, comparisons=(result.comparisons[1], result.comparisons[0]) + result.comparisons[2:]))
        with self.assertRaises(ValueError):
            export_dashboard_antigravity(replace(result, decisions=(replace(result.decisions[0], dataset="ECG5000_normal_1"),)))


if __name__ == "__main__":
    unittest.main()
