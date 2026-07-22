from __future__ import annotations

import unittest

from experiments.exp290_train_raw_exact_normal_guard.independent_reaggregate import reaggregate_rows


class IndependentReaggregateTest(unittest.TestCase):
    def test_reaggregates_baseline_guarded_and_removed_counts(self) -> None:
        rows = [
            {
                "baseline_hard_count": "2", "baseline_hard_tp": "1", "baseline_hard_fp": "1",
                "guarded_hard_count": "1", "guarded_hard_tp": "1", "guarded_hard_fp": "0",
                "train_raw_exact_normal_count": "1", "train_raw_exact_normal_tp": "0", "train_raw_exact_normal_fp": "1",
            },
            {
                "baseline_hard_count": "1", "baseline_hard_tp": "1", "baseline_hard_fp": "0",
                "guarded_hard_count": "1", "guarded_hard_tp": "1", "guarded_hard_fp": "0",
                "train_raw_exact_normal_count": "0", "train_raw_exact_normal_tp": "0", "train_raw_exact_normal_fp": "0",
            },
        ]
        result = reaggregate_rows(rows)
        self.assertEqual(result["datasets"], 2)
        self.assertEqual(result["baseline_hard_count"], 3)
        self.assertEqual(result["guarded_hard_count"], 2)
        self.assertEqual(result["train_raw_exact_normal_count"], 1)
        self.assertEqual(result["tp_reduction"], 0)
        self.assertEqual(result["fp_reduction"], 1)


if __name__ == "__main__":
    unittest.main()
