from __future__ import annotations

import unittest

import numpy as np

from experiments.exp292_standard_review_evidence_tiering.evaluation import evaluate_review_tiers


class EvaluationTests(unittest.TestCase):
    def test_review_metrics_are_separate_from_unchanged_hard_metrics(self) -> None:
        prediction = {
            "dataset": "demo",
            "n_test": 5,
            "automatic_hard_indices": [4],
            "standard_review_indices": [0, 1, 2],
            "review_priority_1_indices": [0],
            "review_priority_2_indices": [1],
            "review_priority_3_indices": [2],
        }
        rows, summary = evaluate_review_tiers([prediction], {"demo": np.asarray([1, 0, 1, 0, 1])})

        self.assertEqual(summary["automatic_hard_count"], 1)
        self.assertEqual(summary["automatic_hard_tp"], 1)
        self.assertEqual(summary["review_priority_1_tp"], 1)
        self.assertEqual(summary["review_priority_2_fp"], 1)
        self.assertEqual(summary["review_priority_3_tp"], 1)
        self.assertTrue(summary["review_metrics_human_assisted_only"])
        self.assertEqual(rows[0]["review_priority_1_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
