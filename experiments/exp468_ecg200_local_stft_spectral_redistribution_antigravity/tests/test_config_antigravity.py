from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class Exp468ConfigTestAntigravity(unittest.TestCase):
    def test_frozen_stft_authority_antigravity(self) -> None:
        """Break caught: the approved S10/C10 identity or STFT geometry drifts."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        c = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(
            (c.experiment, c.method_order, c.primary_candidate),
            ("Exp468", ("C0", "S10", "C10"), "C10"),
        )
        self.assertEqual(c.window_starts, (0, 14, 28, 42, 56))
        self.assertEqual(
            (c.window_length, c.hop_length, c.rfft_length, c.non_dc_bins),
            (28, 14, 28, 14),
        )
        self.assertEqual(
            c.change_types,
            (
                "level_shift",
                "gradual_drift",
                "amplitude_scaling",
                "period_change",
            ),
        )
        self.assertEqual((c.power_epsilon, c.period_auc_min), (1e-12, 0.85))
        self.assertEqual(c.hann_definition, "numpy_hanning_symmetric")
        self.assertEqual(c.js_log_base, "e")
        self.assertFalse(hasattr(c, "mean_auc_min"))
        self.assertFalse(hasattr(c, "backend"))

    def test_frozen_data_and_governance_truth_antigravity(self) -> None:
        """Break caught: TRAIN-only identity, fold geometry, or design digest drifts."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.config_antigravity import (
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
        self.assertEqual(
            c.approved_design_sha256,
            "fa799dd628028163b6bc2dee5af6fd3f507b555849a00b194e32ecdc3f7078b7",
        )

    def test_frozen_config_rejects_mutation_and_type_aliases_antigravity(self) -> None:
        """Break caught: callers replace exact constants with mutable or aliased values."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        c = FROZEN_CONFIG_ANTIGRAVITY
        with self.assertRaises(FrozenInstanceError):
            c.primary_candidate = "S10"  # type: ignore[misc]
        mutations = (
            ("normal_rows", True),
            ("seeds", list(c.seeds)),
            ("window_starts", list(c.window_starts)),
            ("change_types", tuple(reversed(c.change_types))),
            ("method_order", ("C0", "C10", "S10")),
            ("power_epsilon", 0.0),
            ("primary_candidate", "S10"),
            ("period_auc_min", 0.90),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                replace(c, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
