from __future__ import annotations

from dataclasses import replace
import unittest

from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.tests.test_evaluation_antigravity import (
    decisions_antigravity,
    exp456_authority_antigravity,
)


def result_antigravity():
    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.evaluation_antigravity import (
        evaluate_routes_antigravity,
    )

    authority = exp456_authority_antigravity()
    return evaluate_routes_antigravity(authority, decisions_antigravity(authority))


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_exact_aggregate_and_train_gate_allowlist_antigravity(self) -> None:
        """Break caught: dashboard leaks raw TRAIN/TEST evidence or reorders scientific rows."""
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        payload = export_dashboard_antigravity(result_antigravity())
        self.assertEqual(
            tuple(payload),
            (
                "experiment_id",
                "status",
                "passed",
                "ecg_routes",
                "methods",
                "family_metrics",
                "criteria",
            ),
        )
        self.assertEqual(
            tuple(row["dataset"] for row in payload["ecg_routes"]),
            tuple(f"ECG5000_normal_{index}" for index in range(1, 6)),
        )
        self.assertEqual(
            tuple(row["method"] for row in payload["methods"]),
            (
                "always_baseline",
                "family_wide_local_gap",
                "oracle_per_dataset",
                "ecg_train_confidence_router",
            ),
        )
        self.assertEqual(
            tuple(row["family"] for row in payload["family_metrics"]),
            ("ECG5000", "ChlorineConcentration"),
        )
        self.assertEqual(
            tuple(row["metric"] for row in payload["criteria"]),
            ("final_fn", "final_fp", "final_f1"),
        )
        forbidden = {
            "normal_trials",
            "normal_alerts",
            "normal_wilson_upper",
            "pooled_sensitivity",
            "per_type_sensitivity",
            "score",
            "p_value",
            "labels",
            "indices",
            "database",
            "path",
            "sha256",
            "logs",
            "rows",
        }

        def keys_antigravity(value):
            if type(value) is dict:
                yield from value.keys()
                for child in value.values():
                    yield from keys_antigravity(child)
            elif type(value) is list:
                for child in value:
                    yield from keys_antigravity(child)

        self.assertTrue(forbidden.isdisjoint(keys_antigravity(payload)))

    def test_rejects_wrong_orders_types_and_unknown_input_fields_antigravity(self) -> None:
        """Break caught: an unordered/malformed result is normalized into a plausible dashboard."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            ConfusionAntigravity,
        )
        from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.dashboard_export_antigravity import (
            export_dashboard_antigravity,
        )

        result = result_antigravity()
        broken = (
            replace(result, comparisons=tuple(reversed(result.comparisons))),
            replace(result, decisions=tuple(reversed(result.decisions))),
            replace(result, family_metrics=tuple(reversed(result.family_metrics))),
            replace(
                result,
                comparisons=(
                    replace(result.comparisons[0], confusion=ConfusionAntigravity(True, 66, 837, 70788)),
                )
                + result.comparisons[1:],
            ),
        )
        for value in broken + ({"result": result, "raw_scores": []},):
            with self.subTest(value=value), self.assertRaises(ValueError):
                export_dashboard_antigravity(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
