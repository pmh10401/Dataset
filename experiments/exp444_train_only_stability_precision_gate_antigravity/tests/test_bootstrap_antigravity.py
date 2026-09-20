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

    def test_full_1117_population_replays_literal_pcg64_intervals_antigravity(self) -> None:
        """Changing frozen population sampling or macro aggregation must fail these independent literals."""
        baseline = tuple({"dataset": f"D{index:04d}", "tp": 1, "fp": 0, "fn": 1, "tn": 2} for index in range(1117))
        comparator = list(baseline); comparator[0] = {"dataset": "D0000", "tp": 2, "fp": 0, "fn": 0, "tn": 2}
        result = paired_bootstrap_antigravity(baseline, tuple(comparator))
        self.assertEqual(result.sample_indices_sha256, "a1ecb839acc9439ae698388806d15a4610069ac4f388795d2bbf766a3f74888c")
        self.assertAlmostEqual(result.micro_delta, 0.00039777247414485295)
        self.assertAlmostEqual(result.macro_delta, 0.0002984183825723852)
        self.assertEqual(result.micro_interval, (0.0, 0.0011926058437686793))
        self.assertEqual(result.macro_interval, (0.0, 0.0008952551477171555))
