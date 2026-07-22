from __future__ import annotations

import unittest
import inspect

import numpy as np

from experiments.exp298_exp201_current_hard_rebase.core import (
    evaluate_predictions,
    rebase_candidate_row,
    summarize_evaluated,
)
from experiments.exp298_exp201_current_hard_rebase import run_pipeline


def exp201_row(**overrides):
    row = {
        "dataset": "Demo_normal_1",
        "variant": "frozen_exp190_high_standard",
        "n_train": 250,
        "n_test": 5,
        "candidate_indices": [1, 2, 4],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    row.update(overrides)
    return row


def exp292_row(**overrides):
    row = {
        "dataset": "Demo_normal_1",
        "n_test": 5,
        "automatic_hard_indices": [2],
        "standard_review_indices": [1],
        "priority_review_indices": [3],
        "shadow_review_indices": [],
        "research_only_recovery_priority_indices": [],
    }
    row.update(overrides)
    return row


class RebaseTests(unittest.TestCase):
    def test_removes_current_hard_and_classifies_remaining_lane_overlap(self):
        row = rebase_candidate_row(exp201_row(), exp292_row())
        self.assertEqual(row["source_exp201_candidate_indices"], [1, 2, 4])
        self.assertEqual(row["already_in_current_hard_indices"], [2])
        self.assertEqual(row["residual_candidate_indices"], [1, 4])
        self.assertEqual(row["residual_review_overlap_indices"], [1])
        self.assertEqual(row["residual_outside_current_lanes_indices"], [4])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertNotIn("y_test", row)

    def test_rejects_dataset_or_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            rebase_candidate_row(exp201_row(), exp292_row(dataset="Other"))
        with self.assertRaisesRegex(ValueError, "n_test mismatch"):
            rebase_candidate_row(exp201_row(), exp292_row(n_test=6))

    def test_rejects_candidate_outside_test_range(self):
        with self.assertRaisesRegex(ValueError, "outside TEST range"):
            rebase_candidate_row(exp201_row(candidate_indices=[5]), exp292_row())


class EvaluationTests(unittest.TestCase):
    def test_evaluates_added_candidates_and_current_hard_union(self):
        prediction = rebase_candidate_row(exp201_row(), exp292_row())
        evaluated = evaluate_predictions([prediction], {"Demo_normal_1": np.array([0, 1, 1, 0, 0])})[0]
        self.assertEqual((evaluated["residual_tp"], evaluated["residual_fp"]), (1, 1))
        self.assertEqual((evaluated["baseline_hard_tp"], evaluated["baseline_hard_fp"]), (1, 0))
        self.assertEqual((evaluated["counterfactual_tp"], evaluated["counterfactual_fp"]), (2, 1))

    def test_marks_zero_f1_repair(self):
        prediction = rebase_candidate_row(
            exp201_row(candidate_indices=[2]),
            exp292_row(automatic_hard_indices=[0], standard_review_indices=[]),
        )
        evaluated = evaluate_predictions([prediction], {"Demo_normal_1": np.array([0, 0, 1, 0, 0])})[0]
        self.assertEqual(evaluated["zero_f1_before"], 1)
        self.assertEqual(evaluated["zero_f1_after"], 0)
        self.assertEqual(evaluated["zero_f1_fixed"], 1)

    def test_summarizes_frozen_size_strata_and_80pct_reporting_gate(self):
        predictions = [
            rebase_candidate_row(exp201_row(dataset="A", n_train=150, candidate_indices=[1]), exp292_row(dataset="A", automatic_hard_indices=[])),
            rebase_candidate_row(exp201_row(dataset="B", n_train=250, candidate_indices=[1]), exp292_row(dataset="B", automatic_hard_indices=[])),
        ]
        labels = {"A": np.array([0, 1, 0, 0, 0]), "B": np.array([0, 1, 0, 0, 0])}
        summary = summarize_evaluated(evaluate_predictions(predictions, labels))
        all_row = next(row for row in summary if row["size_stratum"] == "100+")
        self.assertEqual(all_row["datasets"], 2)
        self.assertEqual((all_row["residual_tp"], all_row["residual_fp"]), (2, 0))
        self.assertEqual(all_row["retrospective_80pct_point_gate"], 1)


class PipelineBoundaryTests(unittest.TestCase):
    def test_prediction_api_does_not_accept_test_labels(self):
        self.assertNotIn("y_test", inspect.signature(run_pipeline.run_prediction).parameters)
        self.assertNotIn("labels", inspect.signature(run_pipeline.run_prediction).parameters)

    def test_frozen_input_hashes_are_explicit(self):
        self.assertEqual(len(run_pipeline.EXP201_SHA256), 64)
        self.assertEqual(len(run_pipeline.EXP292_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
