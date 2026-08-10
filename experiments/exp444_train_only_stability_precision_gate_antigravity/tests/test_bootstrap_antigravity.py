"""RED/GREEN coverage for the frozen Exp444 paired bootstrap contract."""
from __future__ import annotations

import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.bootstrap_antigravity import (
    paired_bootstrap_antigravity,
)


class PairedBootstrapAntigravityTests(unittest.TestCase):
    def test_paired_bootstrap_uses_frozen_pcg64_contract_antigravity(self) -> None:
        """Changing sample shape, seed, percentile method, or pairing must fail."""
        baseline = (
            {"dataset": "A", "tp": 1, "fp": 1, "fn": 0, "tn": 2},
            {"dataset": "B", "tp": 0, "fp": 0, "fn": 0, "tn": 4},
        )
        comparator = (
            {"dataset": "A", "tp": 2, "fp": 0, "fn": 0, "tn": 2},
            {"dataset": "B", "tp": 0, "fp": 1, "fn": 0, "tn": 3},
        )

        result = paired_bootstrap_antigravity(baseline, comparator)

        self.assertEqual(result.seed, 44420260811)
        self.assertEqual(result.samples, 20_000)
        self.assertEqual(result.sample_dtype, "int64")
        self.assertEqual(result.percentile_method, "linear")
        self.assertEqual(result.baseline_macro_f1, 1.0 / 3.0)
        self.assertEqual(result.comparator_macro_f1, 0.5)
        self.assertAlmostEqual(result.macro_delta, 1.0 / 6.0)
        self.assertAlmostEqual(result.micro_delta, 2.0 / 15.0)
        self.assertEqual(result, paired_bootstrap_antigravity(baseline, comparator))

    def test_zero_denominator_and_dataset_order_are_not_silently_repaired_antigravity(self) -> None:
        """A zero F1 denominator is zero and a broken paired population is rejected."""
        zero = ({"dataset": "A", "tp": 0, "fp": 0, "fn": 0, "tn": 3},)
        self.assertEqual(paired_bootstrap_antigravity(zero, zero).baseline_macro_f1, 0.0)
        with self.assertRaisesRegex(ValueError, "order|paired"):
            paired_bootstrap_antigravity(
                ({"dataset": "A", "tp": 1, "fp": 0, "fn": 0, "tn": 1},),
                ({"dataset": "B", "tp": 1, "fp": 0, "fn": 0, "tn": 1},),
            )
