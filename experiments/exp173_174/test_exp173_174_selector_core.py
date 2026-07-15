from __future__ import annotations

import csv
import unittest
from pathlib import Path

from experiments.exp173_174.selector_core import (
    deterministic_fold,
    select_balanced_train_policy,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "outputs/exp169_170_oob_pooled_calibration"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class BalancedSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.global_rows = read_csv(SOURCE_DIR / "05_exp169_global_summary.csv")
        cls.dataset_rows = read_csv(SOURCE_DIR / "04_exp169_dataset_audit.csv")

    def test_fold_is_deterministic_and_in_range(self) -> None:
        fold = deterministic_fold("demo", folds=5)
        self.assertEqual(fold, deterministic_fold("demo", folds=5))
        self.assertGreaterEqual(fold, 0)
        self.assertLess(fold, 5)
        with self.assertRaisesRegex(ValueError, "at least two"):
            deterministic_fold("demo", folds=1)

    def test_real_train_audit_selects_preregistered_balanced_policy(self) -> None:
        result = select_balanced_train_policy(self.global_rows, self.dataset_rows)
        selected = result["selected"]
        self.assertEqual(result["status"], "policy_selected")
        self.assertEqual(selected["source_config"], "2of3")
        self.assertEqual(selected["pool_policy"], "similarity_pool")
        self.assertEqual(selected["alpha"], 0.01)
        self.assertLessEqual(selected["simultaneous_wilson_upper"], 0.01)
        self.assertEqual(selected["selection_rank"], 1)
        self.assertEqual(selected["fold_count"], 5)
        self.assertEqual(result["test_metrics_used"], 0)

    def test_selector_is_invariant_to_input_order(self) -> None:
        first = select_balanced_train_policy(self.global_rows, self.dataset_rows)
        second = select_balanced_train_policy(
            self.global_rows[::-1], self.dataset_rows[::-1]
        )
        self.assertEqual(first["selected"], second["selected"])

    def test_only_largest_eligible_two_of_three_alpha_is_ranked(self) -> None:
        result = select_balanced_train_policy(self.global_rows, self.dataset_rows)
        ranked = [row for row in result["candidates"] if row["selection_rank"]]
        self.assertTrue(ranked)
        self.assertEqual({row["source_config"] for row in ranked}, {"2of3"})
        self.assertEqual({float(row["alpha"]) for row in ranked}, {0.01})
        self.assertEqual(
            [row["pool_policy"] for row in sorted(ranked, key=lambda row: row["selection_rank"])],
            ["similarity_pool", "coarse_strata_pool", "global_balanced_pool"],
        )

    def test_incomplete_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact 60-row"):
            select_balanced_train_policy(self.global_rows[:-1], self.dataset_rows)

    def test_duplicate_dataset_is_rejected(self) -> None:
        rows = [dict(row) for row in self.dataset_rows]
        target_key = (
            rows[0]["pool_policy"],
            float(rows[0]["alpha"]),
            rows[0]["source_config"],
        )
        matching = [
            row
            for row in rows
            if (row["pool_policy"], float(row["alpha"]), row["source_config"])
            == target_key
        ]
        matching[1]["dataset"] = matching[0]["dataset"]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            select_balanced_train_policy(self.global_rows, rows)

    def test_selector_abstains_when_risk_target_is_too_small(self) -> None:
        result = select_balanced_train_policy(
            self.global_rows,
            self.dataset_rows,
            risk_target=1e-6,
        )
        self.assertEqual(result["status"], "policy_abstention")
        self.assertIsNone(result["selected"])


if __name__ == "__main__":
    unittest.main()
