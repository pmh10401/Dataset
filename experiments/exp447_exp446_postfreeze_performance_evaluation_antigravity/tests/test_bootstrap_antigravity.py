"""Shared-draw paired bootstrap behavior."""
from __future__ import annotations

import unittest


class BootstrapAntigravityTests(unittest.TestCase):
    def _rows_antigravity(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {"dataset": f"dataset-{index:04d}", "tp": 1, "fp": 0, "fn": 1, "tn": 8}
            for index in range(1117)
        )

    def test_shared_draws_produce_replayable_zero_delta_antigravity(self) -> None:
        # Break caught: each comparison silently uses a different RNG draw stream.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.bootstrap_antigravity import (
            bootstrap_draws_antigravity,
            paired_bootstrap_antigravity,
        )

        baseline = self._rows_antigravity()
        draws = bootstrap_draws_antigravity(1117)
        result = paired_bootstrap_antigravity(baseline, baseline, "hard_plus_stable_26", draws=draws)
        replay = paired_bootstrap_antigravity(baseline, baseline, "hard_plus_stable_26", draws=draws)
        self.assertEqual(result.sample_indices_sha256, replay.sample_indices_sha256)
        self.assertEqual(result.sample_indices_sha256, "f960dec37c48599a09d95f7a766d160d133eb9b085536974962bb5412dea993e")
        self.assertEqual(result.micro_delta, 0.0)
        self.assertEqual(result.macro_delta, 0.0)
        self.assertEqual(result.micro_interval, (0.0, 0.0))
        self.assertEqual(result.macro_interval, (0.0, 0.0))

    def test_bootstrap_rejects_order_or_population_drift_antigravity(self) -> None:
        # Break caught: bootstrap pairs different datasets or a partial population.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.bootstrap_antigravity import (
            paired_bootstrap_antigravity,
        )

        baseline = self._rows_antigravity()
        changed_order = tuple(reversed(baseline))
        with self.assertRaises(ValueError):
            paired_bootstrap_antigravity(baseline, changed_order, "hard_plus_source_30")
        with self.assertRaises(ValueError):
            paired_bootstrap_antigravity(baseline[:-1], baseline[:-1], "hard_plus_source_30")


if __name__ == "__main__":
    unittest.main()
