from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from experiments.exp175_176 import independent_reaggregate_exp175_176 as reaggregate


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReaggregationTests(unittest.TestCase):
    def test_reaggregate_reconciles_dataset_and_cause_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(
                root / "04_exp175_dataset_diagnostic.csv",
                [
                    {"dataset": "a", "anomaly_count": 3, "hard_tp": 1, "hard_fp": 1, "hard_fn": 2},
                    {"dataset": "b", "anomaly_count": 1, "hard_tp": 1, "hard_fp": 0, "hard_fn": 0},
                ],
            )
            write_csv(
                root / "05_exp176_missed_anomaly_diagnostic.csv",
                [
                    {"dataset": "a", "candidate_index": 1, "cause_category": "score_visible_policy_miss"},
                    {"dataset": "a", "candidate_index": 2, "cause_category": "representation_blind"},
                ],
            )
            result = reaggregate.reaggregate(root, enforce_expected=False)
            self.assertEqual(result["datasets"], 2)
            self.assertEqual(result["true_anomalies"], 4)
            self.assertEqual(result["hard_misses"], 2)
            self.assertEqual(result["cause_counts"]["score_visible_policy_miss"], 1)
            self.assertTrue(result["all_checks_passed"])


if __name__ == "__main__":
    unittest.main()
