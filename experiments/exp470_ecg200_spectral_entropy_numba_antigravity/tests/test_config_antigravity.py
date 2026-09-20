from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class Exp470ConfigTestAntigravity(unittest.TestCase):
    def test_frozen_spectral_entropy_authority_antigravity(self) -> None:
        """Break caught: Exp470's approved geometry or E11/C11 identity drifts."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.seeds, (20260721, 20261721))
        self.assertEqual(
            (
                config.folds,
                config.normal_rows,
                config.source_width,
                config.interpolated_width,
                config.onset,
                config.window_length,
                config.window_starts,
                config.dft_bins,
                config.power_epsilon,
                config.method_order,
                config.change_types,
                config.required_speedup,
            ),
            (
                5,
                69,
                96,
                140,
                56,
                28,
                (0, 14, 28, 42, 56),
                14,
                1e-12,
                ("C0", "E11", "C11"),
                (
                    "level_shift",
                    "gradual_drift",
                    "amplitude_scaling",
                    "period_change",
                ),
                1.1,
            ),
        )

    def test_frozen_config_rejects_mutation_and_exact_type_aliases_antigravity(self) -> None:
        """Break caught: callers replace frozen authority with reordered or aliased values."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        with self.assertRaises(FrozenInstanceError):
            config.method_order = ("C0", "C11", "E11")  # type: ignore[misc]
        for field_name, value in (
            ("seeds", [20260721, 20261721]),
            ("folds", True),
            ("window_starts", (0, 14, 42, 28, 56)),
            ("dft_bins", 13),
            ("power_epsilon", 0.0),
            ("method_order", ("C0", "C11", "E11")),
            ("change_types", tuple(reversed(config.change_types))),
            ("required_speedup", 1.0),
        ):
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
