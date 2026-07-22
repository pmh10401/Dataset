from __future__ import annotations

import unittest

import numpy as np

from experiments.exp300_independent_evidence_consensus.core import (
    build_consensus_row,
    evaluate_predictions,
    summarize_evaluated,
)


def source(name: str, values: list[int], *, alpha: float = 0.005, dataset: str = "Demo"):
    return {"dataset": dataset, "variant": name, "alpha": alpha, "n_test": 6, "candidate_indices": values}


def baseline(**overrides):
    row = {"dataset": "Demo", "n_train": 250, "n_test": 6, "current_hard_indices": [3]}
    row.update(overrides)
    return row


class ConsensusTests(unittest.TestCase):
    def test_builds_two_and_three_source_consensus_and_removes_hard(self):
        row = build_consensus_row(
            source("multi", [1, 2, 3]),
            source("structural", [2, 3, 4]),
            source("change", [3, 4, 5]),
            baseline(),
        )
        self.assertEqual(row["consensus_2of3_indices"], [2, 3, 4])
        self.assertEqual(row["consensus_3of3_indices"], [3])
        self.assertEqual(row["residual_2of3_indices"], [2, 4])
        self.assertEqual(row["residual_3of3_indices"], [])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_rejects_mismatch_and_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            build_consensus_row(source("m", [], dataset="A"), source("s", [], dataset="B"), source("c", [], dataset="A"), baseline(dataset="A"))
        with self.assertRaisesRegex(ValueError, "alpha mismatch"):
            build_consensus_row(source("m", [], alpha=0.005), source("s", [], alpha=0.01), source("c", [], alpha=0.005), baseline())
        with self.assertRaisesRegex(ValueError, "outside TEST range"):
            build_consensus_row(source("m", [6]), source("s", [6]), source("c", [6]), baseline())


class EvaluationTests(unittest.TestCase):
    def test_evaluates_both_consensus_lanes_without_changing_baseline(self):
        row = build_consensus_row(
            source("multi", [1, 2, 3]), source("structural", [2, 3, 4]), source("change", [3, 4, 5]), baseline()
        )
        evaluated = evaluate_predictions([row], {"Demo": np.array([0, 0, 1, 1, 0, 0])})[0]
        self.assertEqual((evaluated["residual_2of3_tp"], evaluated["residual_2of3_fp"]), (1, 1))
        self.assertEqual((evaluated["residual_3of3_tp"], evaluated["residual_3of3_fp"]), (0, 0))
        self.assertEqual((evaluated["counterfactual_2of3_tp"], evaluated["counterfactual_2of3_fp"]), (2, 1))

    def test_summary_keeps_alpha_and_size_strata_separate(self):
        predictions = []
        labels = {}
        for dataset, alpha, n_train in (("A", 0.005, 150), ("B", 0.01, 250)):
            predictions.append(build_consensus_row(
                source("m", [1], alpha=alpha, dataset=dataset),
                source("s", [1], alpha=alpha, dataset=dataset),
                source("c", [1], alpha=alpha, dataset=dataset),
                baseline(dataset=dataset, n_train=n_train, current_hard_indices=[]),
            ))
            labels[dataset] = np.array([0, 1, 0, 0, 0, 0])
        summary = summarize_evaluated(evaluate_predictions(predictions, labels))
        all_rows = [row for row in summary if row["size_stratum"] == "100+"]
        self.assertEqual({row["alpha"] for row in all_rows}, {0.005, 0.01})
        self.assertTrue(next(row for row in all_rows if row["alpha"] == 0.005)["primary_alpha"])


if __name__ == "__main__":
    unittest.main()
