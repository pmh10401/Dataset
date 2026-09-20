"""Shared-draw paired bootstrap behavior for Exp448."""
from __future__ import annotations

import unittest


class BootstrapAntigravityTests(unittest.TestCase):
    def _rows_antigravity(self) -> tuple[dict[str, object], ...]:
        return tuple({"dataset": f"dataset-{index:04d}", "tp": 1, "fp": 0, "fn": 1, "tn": 8} for index in range(1117))

    def test_shared_draws_produce_replayable_zero_delta_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.bootstrap_antigravity import (
            bootstrap_draws_antigravity, paired_bootstrap_antigravity,
        )

        baseline = self._rows_antigravity()
        draws = bootstrap_draws_antigravity(1117)
        result = paired_bootstrap_antigravity(baseline, baseline, "hard_plus_stable_26", draws=draws)
        self.assertEqual(result.sample_indices_sha256, "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e")
        self.assertEqual((result.micro_delta, result.macro_delta, result.micro_interval, result.macro_interval), (0.0, 0.0, (0.0, 0.0), (0.0, 0.0)))

    def test_bootstrap_rejects_order_population_and_forged_draws_antigravity(self) -> None:
        import numpy as np
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.bootstrap_antigravity import (
            paired_bootstrap_antigravity,
        )

        baseline = self._rows_antigravity()
        with self.assertRaises(ValueError):
            paired_bootstrap_antigravity(baseline, tuple(reversed(baseline)), "hard_plus_source_30")
        with self.assertRaises(ValueError):
            paired_bootstrap_antigravity(baseline[:-1], baseline[:-1], "hard_plus_source_30")
        with self.assertRaises(ValueError):
            paired_bootstrap_antigravity(baseline, baseline, "hard_plus_stable_26", draws=np.zeros((20000, 1117), dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
