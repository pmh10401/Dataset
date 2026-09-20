from __future__ import annotations

import copy
import math
import unittest

from experiments.exp460_ecg_strict_fallback_antigravity.tests.test_artifacts_antigravity import (
    authority_antigravity,
    expected_dashboard_antigravity,
    result_antigravity,
)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_exports_only_allowlisted_scientific_payload_antigravity(self) -> None:
        """Break caught: dashboard leaks raw rows, paths, labels, or omits the fixed Task 4 summary fields."""
        from experiments.exp460_ecg_strict_fallback_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        dashboard = export_dashboard_antigravity(authority_antigravity(), result_antigravity())
        expected = expected_dashboard_antigravity()
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
        self._assert_exact_json_antigravity(dashboard, expected)

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

        mutations = (
            lambda value: value["comparison_rows"][0].__setitem__("tp", False),
            lambda value: value["failed_folds"][0].__setitem__("fold_index", True),
            lambda value: value.__setitem__("worst_period_change_auc", float("nan")),
            lambda value: value.__setitem__("comparison_rows", tuple(value["comparison_rows"])),
            lambda value: value["comparison_rows"][1].__setitem__("f1", "0.5796148426491311"),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                mutated = copy.deepcopy(dashboard)
                mutate(mutated)
                with self.assertRaises(AssertionError):
                    self._assert_exact_json_antigravity(mutated, expected)

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

    def _assert_exact_json_antigravity(self, actual: object, expected: object) -> None:
        self.assertIs(type(actual), type(expected))
        if type(expected) is dict:
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self._assert_exact_json_antigravity(actual[key], expected[key])  # type: ignore[index]
            return
        if type(expected) is list:
            self.assertEqual(len(actual), len(expected))  # type: ignore[arg-type]
            for actual_item, expected_item in zip(actual, expected, strict=True):  # type: ignore[arg-type]
                self._assert_exact_json_antigravity(actual_item, expected_item)
            return
        if type(expected) is float:
            self.assertTrue(math.isfinite(actual))  # type: ignore[arg-type]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
