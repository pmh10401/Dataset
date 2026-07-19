from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

from experiments.exp226_235_hard_alert_quality.baseline_exp137 import BaselineLanes
from experiments.exp236_detection_coverage_expansion.core import (
    build_expanded_lanes,
    freeze_predictions,
    load_verified_predictions,
)
from experiments.exp236_detection_coverage_expansion.run_experiment_236 import predict


class Exp236CoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = BaselineLanes(
            dataset="demo",
            test_size=10,
            hard=frozenset({1, 2}),
            standard_review=frozenset({3}),
            priority_review=frozenset({4}),
            hard_tp=0,
            hard_fp=2,
        )

    def test_three_source_adds_hard_and_exact_two_adds_standard(self) -> None:
        lanes = build_expanded_lanes(
            self.baseline,
            minimum_two={5, 6, 7},
            minimum_three={6, 7},
        )
        self.assertEqual(lanes.hard, frozenset({1, 2, 6, 7}))
        self.assertEqual(lanes.standard_review, frozenset({3, 5}))
        self.assertEqual(lanes.priority_review, frozenset({4}))

    def test_source_three_must_be_subset_of_source_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "subset"):
            build_expanded_lanes(self.baseline, minimum_two={5}, minimum_three={6})

    def test_new_candidates_cannot_overlap_exp137_lanes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exp137"):
            build_expanded_lanes(self.baseline, minimum_two={2, 5}, minimum_three=set())

    def test_prediction_signature_has_no_test_labels(self) -> None:
        signature = inspect.signature(predict)
        self.assertNotIn("y_test", signature.parameters)
        self.assertNotIn("load_test_labels", inspect.getsource(predict).lower())

    def test_frozen_prediction_rejects_tampering(self) -> None:
        rows = [{
            "experiment_id": "experiment_236_three_source_hard_two_source_review",
            "dataset": "demo",
            "n_test": 10,
            "hard_indices": [1],
            "standard_review_indices": [2],
            "priority_review_indices": [3],
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction, manifest = freeze_predictions(rows, root)
            prediction.write_text(prediction.read_text() + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_verified_predictions(prediction, manifest)

    def test_dashboard_catalog_marks_exp236_as_research_only(self) -> None:
        from serve_rank_dashboard import EXPERIMENTS

        row = EXPERIMENTS["experiment_236_three_source_hard_two_source_review"]
        self.assertFalse(row["comparison_eligible"])
        self.assertIn("prior_test", row["scope"])


if __name__ == "__main__":
    unittest.main()
