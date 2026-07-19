from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp249_250_official_mini_multirocket.independent_reaggregate import (
    compare_summary,
)


class IndependentReaggregationTest(unittest.TestCase):
    def test_matching_summary_passes_and_tampering_fails(self) -> None:
        result_rows = [
            {
                "variant": "v",
                "alpha": "0.01",
                "candidate_tp": "2",
                "candidate_fp": "1",
                "candidate_f1": "0.8",
            }
        ]
        matching = [
            {
                "variant": "v",
                "alpha": "0.01",
                "candidate_count": "3",
                "candidate_tp": "2",
                "candidate_fp": "1",
                "candidate_precision": str(2 / 3),
                "candidate_mean_f1": "0.8",
            }
        ]
        self.assertEqual(compare_summary(result_rows, matching, views=("candidate",)), [])
        matching[0]["candidate_fp"] = "2"
        self.assertTrue(compare_summary(result_rows, matching, views=("candidate",)))


if __name__ == "__main__":
    unittest.main()
