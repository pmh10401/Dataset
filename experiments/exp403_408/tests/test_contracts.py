import math
import unittest
from decimal import Decimal

import numpy as np

from experiments.exp403_408.contracts import (
    ALPHAS,
    EXPECTED_DATASETS,
    EXPECTED_HARD,
    EXPECTED_PRIORITY,
    EXPECTED_STANDARD,
    SEED,
    train_eligibility,
    validate_prediction_row,
)


def valid_prediction_row() -> dict[str, object]:
    return {
        "dataset": "synthetic-001",
        "candidate_index": 3,
        "scores_by_alpha": {str(alpha): 0.25 for alpha in ALPHAS},
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


class ContractTests(unittest.TestCase):
    def test_frozen_constants(self):
        self.assertEqual(SEED, 20260727)
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(
            (EXPECTED_DATASETS, EXPECTED_HARD, EXPECTED_STANDARD, EXPECTED_PRIORITY),
            (1117, 2218, 806, 31),
        )

    def test_train_eligibility_boundaries(self):
        self.assertEqual(train_eligibility(4).level, "abstain")
        self.assertEqual(train_eligibility(5).level, "robust_distance")
        self.assertEqual(train_eligibility(20).level, "classical_committee")
        self.assertEqual(train_eligibility(50).level, "learned_methods")

    def test_prediction_rejects_test_metrics(self):
        row = valid_prediction_row()
        row["hard_fp"] = 1
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_prediction_row(row)

    def test_prediction_rejects_nested_test_metrics(self):
        row = valid_prediction_row()
        row["diagnostics"] = {"nested_precision": 0.5}
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_prediction_row(row)

    def test_prediction_rejects_nested_test_label_and_metrics(self):
        for field in ("test_label", "test_accuracy", "test_auc"):
            row = valid_prediction_row()
            row["diagnostics"] = {field: 0.5}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "forbidden"):
                validate_prediction_row(row)

    def test_prediction_allows_contains_test_labels_declaration(self):
        validate_prediction_row(valid_prediction_row())

    def test_prediction_requires_all_four_alphas(self):
        row = valid_prediction_row()
        del row["scores_by_alpha"]["0.05"]
        with self.assertRaisesRegex(ValueError, "alpha"):
            validate_prediction_row(row)

    def test_prediction_rejects_nonfinite_scores(self):
        row = valid_prediction_row()
        row["scores_by_alpha"]["0.005"] = math.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_row(row)

    def test_prediction_rejects_nonfinite_nested_numbers(self):
        row = valid_prediction_row()
        row["diagnostics"] = {"score_spread": math.nan}
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_row(row)

    def test_prediction_rejects_nonfinite_decimal(self):
        row = valid_prediction_row()
        row["diagnostics"] = {"score_spread": Decimal("NaN")}
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_row(row)

    def test_prediction_rejects_nonfinite_numpy_scalar(self):
        row = valid_prediction_row()
        row["diagnostics"] = {"score_spread": np.float32(np.nan)}
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_prediction_row(row)


if __name__ == "__main__":
    unittest.main()
