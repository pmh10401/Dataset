import unittest

from experiments.exp263_264_full_missing_aware_integration import config


class ConfigurationContractTest(unittest.TestCase):
    def test_policy_constants_are_preregistered(self) -> None:
        self.assertEqual(config.ALPHA, 0.005)
        self.assertEqual(config.MIN_TRAIN_FOR_PROMOTION, 100)
        self.assertEqual(config.REQUIRED_PROMOTION_AGREEMENT, 3)
        self.assertEqual(config.MASK_VARIANT, "mask_025__seed_intersection")


if __name__ == "__main__":
    unittest.main()
