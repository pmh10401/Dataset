from __future__ import annotations

import unittest

from experiments.exp309_316_fn_recovery.neural_common import (
    choose_device, train_only_split, train_validation_calibration_split,
)


class NeuralCommonTests(unittest.TestCase):
    def test_auto_prefers_mps(self):
        self.assertEqual(choose_device("auto", mps_available=True).type, "mps")
        self.assertEqual(choose_device("auto", mps_available=False).type, "cpu")

    def test_train_split_is_deterministic_and_complete(self):
        fit, calibration = train_only_split(20, seed=20260722)
        self.assertTrue(set(fit).isdisjoint(calibration))
        self.assertEqual(sorted(fit + calibration), list(range(20)))
        self.assertEqual((fit, calibration), train_only_split(20, seed=20260722))

    def test_three_way_split_keeps_calibration_out_of_early_stopping(self):
        fit, validation, calibration = train_validation_calibration_split(20, seed=20260722)
        self.assertFalse(set(fit) & set(validation))
        self.assertFalse(set(fit) & set(calibration))
        self.assertFalse(set(validation) & set(calibration))
        self.assertEqual(sorted(fit + validation + calibration), list(range(20)))


if __name__ == "__main__": unittest.main()
