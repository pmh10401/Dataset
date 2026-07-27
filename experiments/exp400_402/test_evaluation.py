from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp400_402.artifact_io import PREDICTION_NAME, write_prediction
from experiments.exp400_402.evaluation import evaluate_prediction, evaluate_rows
from experiments.exp400_402.prediction import build_shadow_predictions
from experiments.exp400_402.test_prediction import exp292_fixture, exp397_fixture


def labels_fixture() -> dict[str, list[str]]:
    return {
        "A": ["0", "0", "0", "0", "0", "0"],
        "B": ["1", "0", "0", "0"],
        "C": ["0", "0", "0"],
    }


class EvaluationTests(unittest.TestCase):
    def test_fixed_workload_compares_frozen_ranks(self) -> None:
        predictions = build_shadow_predictions(exp292_fixture(), exp397_fixture())
        candidate_rows, summary_rows, _ = evaluate_rows(
            predictions,
            labels_fixture(),
            workloads=(1, 2),
        )
        self.assertTrue(candidate_rows)
        baseline = next(
            row
            for row in summary_rows
            if row["variant"] == "exp292_baseline"
            and row["alpha"] == 0.005
            and row["workload_type"] == "global_k"
            and row["workload_k"] == 1
        )
        shadow = next(
            row
            for row in summary_rows
            if row["variant"] == "tier_preserving_grammar"
            and row["alpha"] == 0.005
            and row["workload_type"] == "global_k"
            and row["workload_k"] == 1
        )
        self.assertEqual(baseline["precision_at_k"], 0.0)
        self.assertEqual(shadow["precision_at_k"], 1.0)
        self.assertFalse(shadow["test_metrics_select_policy"])

    def test_evaluation_refuses_modified_prediction_before_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = build_shadow_predictions(
                exp292_fixture(),
                exp397_fixture(),
            )
            write_prediction(root, predictions, full_run=False)
            with (root / PREDICTION_NAME).open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            called = False

            def forbidden_loader(_datasets: set[str]) -> dict[str, list[str]]:
                nonlocal called
                called = True
                raise AssertionError("labels must not be read")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                evaluate_prediction(root, label_loader=forbidden_loader)
            self.assertFalse(called)


if __name__ == "__main__":
    unittest.main()
