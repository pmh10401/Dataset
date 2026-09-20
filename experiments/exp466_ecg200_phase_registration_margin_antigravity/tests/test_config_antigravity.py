from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class Exp466ConfigTestAntigravity(unittest.TestCase):
    def test_frozen_config_antigravity(self) -> None:
        """Break caught: the approved C9 identity or fixed gates drift."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        c = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(
            (c.experiment, c.method_order, c.primary_candidate),
            ("Exp466", ("C0", "P9", "C9"), "C9"),
        )
        self.assertEqual((c.warp_factor, c.shared_length), (1.25, 67))
        self.assertEqual((c.per_type_auc_min, c.mean_auc_min), (0.85, 0.90))

    def test_frozen_data_and_governance_truth_antigravity(self) -> None:
        """Break caught: TRAIN-only shape, fixed fold geometry, or zero-use truth drifts."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        c = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual((c.dataset, c.data_split), ("ECG200", "TRAIN_only"))
        self.assertEqual(c.train_shape, (100, 96))
        self.assertEqual(c.normal_rows, 69)
        self.assertEqual(c.train_md5, "808875aeac1c358a90c630fe369327c4")
        self.assertEqual(c.interpolated_width, 140)
        self.assertEqual((c.seeds, c.folds), ((20260721, 20261721), 5))
        self.assertEqual((c.onset, c.suffix_length), (56, 84))
        self.assertEqual(c.required_fold_results, 10)
        self.assertEqual(c.normal_trials_per_seed, 69)
        self.assertEqual(c.label_query_count_must_remain, 0)
        self.assertEqual(c.official_evaluate_count_must_remain, 0)
        self.assertEqual(c.official_inspect_count_must_remain, 0)
        self.assertEqual(
            c.approved_design_sha256,
            "eddb057724e0ef21402c07b9f362bc6b260fc917a9f6bfdcb130f118b30a5c66",
        )

    def test_frozen_config_rejects_mutation_and_type_aliases_antigravity(self) -> None:
        """Break caught: callers replace immutable tuples, exact floats, or zero counts."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        c = FROZEN_CONFIG_ANTIGRAVITY
        with self.assertRaises(FrozenInstanceError):
            c.primary_candidate = "P9"  # type: ignore[misc]
        mutations = (
            ("normal_rows", True),
            ("seeds", list(c.seeds)),
            ("method_order", ("C0", "C9", "P9")),
            ("warp_factor", 1.20),
            ("shared_length", 66),
            ("primary_candidate", "P9"),
            ("label_query_count_must_remain", 1),
            ("official_evaluate_count_must_remain", 1),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                replace(c, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
