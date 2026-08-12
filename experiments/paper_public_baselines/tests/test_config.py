from __future__ import annotations

import dataclasses
import unittest

from experiments.paper_public_baselines.config import (
    ALPHAS,
    CLASSICAL_METHOD_IDS,
    DEEP_METHOD_IDS,
    EXPECTED_DATASETS,
    EXPECTED_DB_SHA256,
    FROZEN_CONFIG,
    METHOD_IDS,
    SEED,
)


class ConfigTests(unittest.TestCase):
    def test_frozen_public_baseline_contract(self) -> None:
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02))
        self.assertEqual(SEED, 20260812)
        self.assertEqual(EXPECTED_DATASETS, 1117)
        self.assertEqual(
            EXPECTED_DB_SHA256,
            "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
        )
        self.assertEqual(METHOD_IDS, CLASSICAL_METHOD_IDS + DEEP_METHOD_IDS)
        self.assertEqual(len(METHOD_IDS), 6)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            FROZEN_CONFIG.seed = 1


if __name__ == "__main__":
    unittest.main()
