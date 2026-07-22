from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.exp289_label_consistency_audit.audit import (
    exact_cross_label_groups,
    select_frozen_candidates,
    summarize_candidate_tiers,
    validate_candidate_index,
    write_rows,
)


def _prediction(dataset: str, count: int) -> dict[str, object]:
    records = []
    for index in range(count):
        records.append(
            {
                "candidate_index": index,
                "source_lane": "recovery_priority",
                "variant_pass": {"synthetic_shape_alpha_005": True},
                "synthetic_shape_score": 0.5,
                "pvalue": 0.01,
            }
        )
    return {"dataset": dataset, "candidate_records": records}


class AuditContractTest(unittest.TestCase):
    def test_frozen_population_requires_exactly_43_candidates(self) -> None:
        rows = [_prediction("dataset-a", 43)]
        selected = select_frozen_candidates(rows)
        self.assertEqual(len(selected), 43)
        with self.assertRaisesRegex(ValueError, "43"):
            select_frozen_candidates([_prediction("dataset-a", 42)])

    def test_candidate_index_must_be_in_test_range(self) -> None:
        validate_candidate_index("dataset-a", 2, 3)
        with self.assertRaisesRegex(IndexError, "dataset-a"):
            validate_candidate_index("dataset-a", 3, 3)

    def test_exact_conflicts_group_all_indices_without_pair_explosion(self) -> None:
        normal = np.asarray([[1.0, 2.0], [1.0, 2.0], [9.0, 9.0]], dtype=np.float32)
        anomaly = np.asarray([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
        groups = exact_cross_label_groups(normal, anomaly, normal_indices=[1, 3, 8], anomaly_indices=[2, 7])
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["normal_indices"], "[1,3]")
        self.assertEqual(groups[0]["anomaly_indices"], "[2]")

    def test_tier_summary_is_mutually_exclusive(self) -> None:
        rows = [
            {"outcome": "TP", "train_tier": "raw_exact", "opposite_test_tier": "distinct"},
            {"outcome": "FP", "train_tier": "near_duplicate", "opposite_test_tier": "raw_exact"},
        ]
        summary = summarize_candidate_tiers(rows)
        self.assertEqual(summary[("TP", "train", "raw_exact")], 1)
        self.assertEqual(summary[("FP", "train", "near_duplicate")], 1)
        self.assertEqual(summary[("FP", "opposite_test", "raw_exact")], 1)
        self.assertEqual(sum(value for key, value in summary.items() if key[0] == "TP"), 2)

    def test_empty_diagnostic_csv_keeps_its_contract_header(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            write_rows(path, [], fieldnames=("dataset", "match_space"))
            self.assertEqual(path.read_text(encoding="utf-8"), "dataset,match_space\n")


if __name__ == "__main__":
    unittest.main()
