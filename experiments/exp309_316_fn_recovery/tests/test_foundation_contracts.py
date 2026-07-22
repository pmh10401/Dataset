from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.preprocessing import preprocessing_contract
from experiments.exp309_316_fn_recovery.calibration import empirical_pvalues
from experiments.exp309_316_fn_recovery.config import CANDIDATE_GRAIN
from experiments.exp309_316_fn_recovery.contracts import freeze_prediction_rows
from experiments.exp309_316_fn_recovery.foundation import (
    deterministic_folds,
    select_frozen_evaluation_universe,
)


class FoundationContractTests(unittest.TestCase):
    def test_preprocessing_contract_and_candidate_grain(self):
        self.assertEqual(preprocessing_contract()["order"], [
            "missing_repair", "train_length_alignment", "normalization",
        ])
        self.assertEqual(CANDIDATE_GRAIN, "test_instance_virtual_run")

    def test_empirical_pvalues_are_monotone(self):
        actual = empirical_pvalues(np.array([1.0, 2.0, 3.0]), np.array([0.0, 4.0]))
        np.testing.assert_allclose(actual, [1.0, 0.25])

    def test_prediction_contract_rejects_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "contains_test_labels"):
                freeze_prediction_rows(
                    [{"dataset": "D", "tp": 1}], root / "p.jsonl", root / "m.json",
                    experiment_id="experiment_test",
                )

    def test_folds_are_deterministic_and_disjoint(self):
        first = deterministic_folds(11, seed=7)
        second = deterministic_folds(11, seed=7)
        self.assertEqual([part.tolist() for part in first], [part.tolist() for part in second])
        flattened = np.concatenate(first)
        self.assertEqual(sorted(flattened.tolist()), list(range(11)))

    def test_frozen_evaluation_universe_preserves_order_and_reports_db_only_rows(self):
        selected, excluded = select_frozen_evaluation_universe(
            ["small", "new_db_only", "large"], {"small", "large"}, expected=2,
        )
        self.assertEqual(selected, ["small", "large"])
        self.assertEqual(excluded, ["new_db_only"])

    def test_frozen_evaluation_universe_rejects_missing_dataset(self):
        with self.assertRaisesRegex(ValueError, "missing from database"):
            select_frozen_evaluation_universe(["small"], {"small", "missing"}, expected=2)


if __name__ == "__main__":
    unittest.main()
