from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.evaluation import (
    PredictionHashError,
    evaluate_frozen_predictions,
)
from experiments.exp267_273_zero_support_fn_recovery.independent_reaggregate import reaggregate
from experiments.exp267_273_zero_support_fn_recovery.promotion_audit import audit_promotion
from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import freeze_recovery_predictions


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.prediction, self.manifest = _fixture(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_evaluation_rejects_tampered_prediction_before_labels(self) -> None:
        self.prediction.write_text(self.prediction.read_text() + " ", encoding="utf-8")
        called = False

        def labels(_expected):
            nonlocal called
            called = True
            return {"d": {0}}

        with self.assertRaises(PredictionHashError):
            evaluate_frozen_predictions(
                self.prediction, self.manifest, expected_datasets=("d",), truth_loader=labels,
            )
        self.assertFalse(called)

    def test_autonomous_and_human_assisted_metrics_are_separate(self) -> None:
        result = evaluate_frozen_predictions(
            self.prediction, self.manifest, expected_datasets=("d",),
            truth_loader=lambda _expected: {"d": {0, 4, 5}}, output_dir=self.root,
        )
        summary = result["summary"]
        self.assertEqual(summary["autonomous"]["hard_tp"], 1)
        self.assertEqual(summary["human_assisted_review"]["review_tp"], 2)
        self.assertNotIn("combined_f1", summary["autonomous"])
        self.assertNotIn("hard_tp", summary["human_assisted_review"])
        self.assertTrue(summary["human_assisted_diagnostic"]["human_assisted_only"])
        self.assertEqual(summary["autonomous"]["hard_alert_delta_from_exp253"], 0)

    def test_promotion_requires_all_frozen_gates(self) -> None:
        audit = audit_promotion(
            independent_groups=2, train_u95=0.004, precision=0.81,
            overall_precision=0.82, invariant=True,
        )
        self.assertTrue(audit.eligible_for_future_experiment)
        self.assertFalse(audit_promotion(
            independent_groups=2, train_u95=0.006, precision=0.81,
            overall_precision=0.82, invariant=True,
        ).eligible_for_future_experiment)

    def test_independent_reaggregation_matches_saved_summary(self) -> None:
        evaluated = evaluate_frozen_predictions(
            self.prediction, self.manifest, expected_datasets=("d",),
            truth_loader=lambda _expected: {"d": {0, 4, 5}}, output_dir=self.root,
        )
        rebuilt = reaggregate(
            evaluated["results_path"], evaluated["summary_path"],
            expected_datasets=1, expected_total_anomalies=3, expected_hard_alerts=1,
        )
        self.assertEqual(rebuilt["datasets"], 1)
        self.assertEqual(rebuilt["hard_tp"], 1)
        self.assertEqual(rebuilt["review_tp"], 2)


def _fixture(root: Path) -> tuple[Path, Path]:
    row = {
        "dataset": "d", "n_train": 10, "n_test": 8,
        "hard_indices": [0], "shadow_indices": [1], "priority_indices": [2],
        "standard_indices": [3], "recovery_shadow_indices": [4],
        "recovery_priority_indices": [5], "data_quality_review_indices": [7],
        "quality_flag_indices": [7], "one_source_sensitivity_indices": [6],
        "support_by_index": {
            "4": ["a", "b", "c"], "5": ["a", "b"], "6": ["a"], "7": ["a", "b", "c"],
        },
        "source_eligibility": {"a": True}, "input_sha256": {"fixture": "hash"},
        "config_sha256": "config", "contains_test_labels": False,
        "contains_evaluation_metrics": False, "review_only": True, "retrospective": True,
    }
    return freeze_recovery_predictions(
        [row], output_dir=root, expected_datasets=("d",), membership_sha256="membership",
        input_sha256={"fixture": "hash"},
    )


if __name__ == "__main__":
    unittest.main()
