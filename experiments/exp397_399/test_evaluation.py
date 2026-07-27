from __future__ import annotations

import inspect
import unittest

from experiments.exp397_399.evaluation import (
    precision_at_k,
    run_evaluation,
)


class EvaluationTests(unittest.TestCase):
    def test_precision_at_k_uses_first_k_candidates(self):
        ranked = [
            {"rank": 1, "is_true_anomaly": True},
            {"rank": 2, "is_true_anomaly": False},
        ]

        result = precision_at_k(ranked, 1, total_review_tp=1)

        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["tp_at_k"], 1)
        self.assertEqual(result["fp_at_k"], 0)
        self.assertEqual(result["precision_at_k"], 1.0)

    def test_precision_at_k_handles_workload_larger_than_universe(self):
        result = precision_at_k(
            [{"rank": 1, "is_true_anomaly": False}],
            25,
            total_review_tp=0,
        )

        self.assertEqual(result["reviewed"], 1)
        self.assertEqual(result["precision_at_k"], 0.0)
        self.assertEqual(result["review_tp_capture_rate"], 0.0)

    def test_evaluation_verifies_prediction_before_labels(self):
        source = inspect.getsource(run_evaluation)

        self.assertLess(
            source.index("load_verified_ranking"),
            source.index("load_test_labels"),
        )


if __name__ == "__main__":
    unittest.main()
