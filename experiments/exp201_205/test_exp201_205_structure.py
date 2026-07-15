from __future__ import annotations

import inspect
import unittest

from experiments.exp201_205 import artifacts


class Exp201205StructureTests(unittest.TestCase):
    def test_prediction_loader_has_no_test_label_argument(self):
        self.assertNotIn("y_test", inspect.signature(artifacts.load_frozen_inputs).parameters)

    def test_evaluation_label_loader_is_separate(self):
        self.assertTrue(callable(artifacts.load_test_labels_for_evaluation))

    def test_stage_ids_are_frozen(self):
        from experiments.exp201_205.run_experiment_201_205 import STAGE_META

        self.assertEqual(list(STAGE_META), ["201", "202", "203", "204", "205"])


if __name__ == "__main__":
    unittest.main()
