from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp299_multirocket_capacity_consensus.core import (
    build_consensus_row,
    evaluate_predictions,
    summarize_evaluated,
)
from experiments.exp299_multirocket_capacity_consensus import run_pipeline


def block(variant: str, candidates: list[int], *, alpha: float = 0.005, dataset: str = "Demo"):
    return {
        "dataset": dataset,
        "variant": variant,
        "alpha": alpha,
        "n_test": 6,
        "candidate_indices": candidates,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def baseline(**overrides):
    row = {
        "dataset": "Demo",
        "n_train": 250,
        "n_test": 6,
        "current_hard_indices": [2],
    }
    row.update(overrides)
    return row


class ConsensusTests(unittest.TestCase):
    def test_intersects_capacity_blocks_and_excludes_current_hard(self):
        row = build_consensus_row(
            block("kernels_512_seed_intersection", [1, 2, 3]),
            block("kernels_1024_seed_intersection", [2, 3, 4]),
            baseline(),
        )
        self.assertEqual(row["block_512_candidate_indices"], [1, 2, 3])
        self.assertEqual(row["block_1024_candidate_indices"], [2, 3, 4])
        self.assertEqual(row["capacity_consensus_indices"], [2, 3])
        self.assertEqual(row["residual_consensus_indices"], [3])
        self.assertEqual(row["residual_block_512_indices"], [1, 3])
        self.assertEqual(row["residual_block_1024_indices"], [3, 4])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_rejects_mismatched_dataset_alpha_or_range(self):
        with self.assertRaisesRegex(ValueError, "dataset mismatch"):
            build_consensus_row(block("kernels_512_seed_intersection", [], dataset="A"), block("kernels_1024_seed_intersection", [], dataset="B"), baseline(dataset="A"))
        with self.assertRaisesRegex(ValueError, "alpha mismatch"):
            build_consensus_row(block("kernels_512_seed_intersection", [], alpha=0.005), block("kernels_1024_seed_intersection", [], alpha=0.01), baseline())
        with self.assertRaisesRegex(ValueError, "outside TEST range"):
            build_consensus_row(block("kernels_512_seed_intersection", [6]), block("kernels_1024_seed_intersection", [6]), baseline())


class EvaluationTests(unittest.TestCase):
    def test_reports_component_consensus_and_union_metrics(self):
        prediction = build_consensus_row(
            block("kernels_512_seed_intersection", [1, 2, 3]),
            block("kernels_1024_seed_intersection", [2, 3, 4]),
            baseline(),
        )
        row = evaluate_predictions([prediction], {"Demo": np.array([0, 1, 1, 1, 0, 0])})[0]
        self.assertEqual((row["block_512_residual_tp"], row["block_512_residual_fp"]), (2, 0))
        self.assertEqual((row["block_1024_residual_tp"], row["block_1024_residual_fp"]), (1, 1))
        self.assertEqual((row["consensus_residual_tp"], row["consensus_residual_fp"]), (1, 0))
        self.assertEqual((row["counterfactual_tp"], row["counterfactual_fp"]), (2, 0))

    def test_summary_keeps_primary_and_sensitivity_alpha_separate(self):
        predictions = []
        labels = {}
        for dataset, alpha in (("A", 0.005), ("B", 0.01)):
            predictions.append(build_consensus_row(
                block("kernels_512_seed_intersection", [1], alpha=alpha, dataset=dataset),
                block("kernels_1024_seed_intersection", [1], alpha=alpha, dataset=dataset),
                baseline(dataset=dataset),
            ))
            labels[dataset] = np.array([0, 1, 0, 0, 0, 0])
        summary = summarize_evaluated(evaluate_predictions(predictions, labels))
        self.assertEqual({row["alpha"] for row in summary}, {0.005, 0.01})
        self.assertTrue(next(row for row in summary if row["alpha"] == 0.005)["primary_alpha"])
        self.assertFalse(next(row for row in summary if row["alpha"] == 0.01)["primary_alpha"])


class BoundaryTests(unittest.TestCase):
    def test_prediction_does_not_accept_labels(self):
        self.assertNotIn("labels", inspect.signature(run_pipeline.run_prediction).parameters)
        self.assertNotIn("y_test", inspect.signature(run_pipeline.run_prediction).parameters)


if __name__ == "__main__":
    unittest.main()
