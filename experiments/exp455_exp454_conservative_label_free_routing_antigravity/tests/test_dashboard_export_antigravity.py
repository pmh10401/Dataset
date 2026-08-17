from __future__ import annotations

import unittest


class DashboardExportAntigravityTests(unittest.TestCase):
    def test_exports_aggregate_only_payload_antigravity(self) -> None:
        """Break caught: dashboard export leaks a dataset-level field or omits a comparison."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.dashboard_export_antigravity import export_aggregate_dashboard_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity

        payload = export_aggregate_dashboard_antigravity(evaluate_routing_antigravity(*RoutingEvaluationAntigravityTests._inputs_antigravity()))
        self.assertEqual(set(payload), {"experiment_id", "status", "completed_folds", "methods", "family_fn_delta", "passed"})
        self.assertEqual(tuple(payload["methods"][0]), ("method", "fn", "fp", "f1"))
        self.assertEqual(len(payload["methods"]), 4)
        self.assertEqual(len(payload["family_fn_delta"]), 2)
        self.assertEqual(payload["completed_folds"], 8)


if __name__ == "__main__":
    unittest.main()
