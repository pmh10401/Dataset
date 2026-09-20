from __future__ import annotations

import inspect
import unittest

from experiments.exp400_402.contracts import prediction_rows_sha256
from experiments.exp400_402.prediction import build_shadow_predictions
from experiments.exp400_402.test_prediction import exp292_fixture, exp397_fixture


class InvarianceTests(unittest.TestCase):
    def test_prediction_signature_has_no_labels(self) -> None:
        parameters = inspect.signature(build_shadow_predictions).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_dataset_input_order_does_not_change_output(self) -> None:
        left = build_shadow_predictions(exp292_fixture(), exp397_fixture())
        right = build_shadow_predictions(
            list(reversed(exp292_fixture())),
            list(reversed(exp397_fixture())),
        )
        self.assertEqual(left, right)
        self.assertEqual(
            prediction_rows_sha256(left),
            prediction_rows_sha256(right),
        )


if __name__ == "__main__":
    unittest.main()
