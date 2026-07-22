from __future__ import annotations

import unittest

import numpy as np

from experiments.exp317_318_train_reliability_fusion.evaluation import evaluate_candidate_set


class EvaluationTests(unittest.TestCase):
    def test_candidate_metrics_keep_shadow_separate_from_hard(self):
        result = evaluate_candidate_set(
            candidates={2, 3, 5},
            truth={2, 4},
            hard={4},
            existing_review={1},
            exp316_candidates={3, 5},
        )
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 2)
        self.assertEqual(result["outside_hard_tp"], 1)
        self.assertEqual(result["outside_all_existing_tp"], 1)
        self.assertEqual(result["exp316_overlap_count"], 2)
        self.assertNotIn("automatic_hard_tp", result)


if __name__ == "__main__":
    unittest.main()
