from __future__ import annotations

import unittest

from experiments.exp386_388.historical_runtime import (
    ALPHAS,
    alpha_tag,
)


class HistoricalRuntimeTests(unittest.TestCase):
    def test_alpha_grid_is_frozen(self) -> None:
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05, 0.10, 0.20))

    def test_alpha_tags_are_stable(self) -> None:
        self.assertEqual(alpha_tag(0.005), "0p005")
        self.assertEqual(alpha_tag(0.20), "0p200")


if __name__ == "__main__":
    unittest.main()
