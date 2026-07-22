import unittest

from experiments.exp278_281_train_only_recovery.contracts import (
    ALPHA, BOOTSTRAP_SEEDS, RISK_TARGETS, VARIANTS,
)


class RiskContractTest(unittest.TestCase):
    def test_preregistered_constants_are_frozen(self) -> None:
        self.assertEqual(ALPHA, 0.005)
        self.assertEqual(RISK_TARGETS, (0.005, 0.01, 0.02))
        self.assertEqual(BOOTSTRAP_SEEDS, tuple(range(20260722, 20260727)))
        self.assertEqual(VARIANTS["risk_001_stability_080"], (0.01, 0.8))


if __name__ == "__main__":
    unittest.main()
