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

    def test_rejects_forged_method_family_or_scalar_types_antigravity(self) -> None:
        """Break caught: a dashboard can publish an unapproved method, dataset-like name, or loose scalar type."""
        from dataclasses import replace
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import ComparisonMetricAntigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.dashboard_export_antigravity import export_aggregate_dashboard_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests

        evaluation = evaluate_routing_antigravity(*RoutingEvaluationAntigravityTests._inputs_antigravity())
        bad_name = replace(evaluation, comparisons=(ComparisonMetricAntigravity("ECG5000__row", evaluation.comparisons[0].confusion, evaluation.comparisons[0].f1),) + evaluation.comparisons[1:])
        bad_family = replace(evaluation, family_fn_delta=(("ECG5000__row", -20),) + evaluation.family_fn_delta[1:])
        bad_bool = replace(evaluation, passed=1)
        for broken in (bad_name, bad_family, bad_bool):
            with self.assertRaises(ValueError):
                export_aggregate_dashboard_antigravity(broken)


if __name__ == "__main__":
    unittest.main()
