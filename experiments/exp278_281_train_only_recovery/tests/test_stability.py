import unittest

import numpy as np

from experiments.exp278_281_train_only_recovery.stability import bootstrap_recurrence


class StabilityTest(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_bounded(self) -> None:
        reference = np.linspace(0.0, 1.0, 250)
        first = bootstrap_recurrence(reference, 2.0, identity="demo")
        second = bootstrap_recurrence(reference, 2.0, identity="demo")
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_extreme_score_is_stable_at_available_resolution(self) -> None:
        reference = np.linspace(0.0, 1.0, 250)
        self.assertEqual(bootstrap_recurrence(reference, 2.0, identity="extreme"), 1.0)


if __name__ == "__main__":
    unittest.main()
