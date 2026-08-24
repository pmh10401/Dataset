from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest


class Exp471ConfigTestAntigravity(unittest.TestCase):
    def test_frozen_acf_peak_lag_authority_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(
            (
                config.seeds,
                config.folds,
                config.normal_rows,
                config.source_width,
                config.interpolated_width,
                config.onset,
                config.warp_factor,
                config.early_start,
                config.late_start,
                config.window_length,
                config.acf_lag_min,
                config.acf_lag_max,
                config.peak_lag_min,
                config.peak_lag_max,
                config.mad_scale,
                config.method_order,
                config.change_types,
                config.required_speedup,
            ),
            (
                (20260721, 20261721),
                5,
                69,
                96,
                140,
                56,
                1.25,
                0,
                84,
                56,
                1,
                25,
                2,
                24,
                1.4826,
                ("C0", "E12", "C12"),
                ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"),
                1.1,
            ),
        )

    def test_frozen_config_rejects_mutation_reordered_tuples_and_exact_type_aliases_antigravity(self) -> None:
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        with self.assertRaises(FrozenInstanceError):
            config.method_order = ("C0", "C12", "E12")  # type: ignore[misc]
        cases = (
            ("seeds", [20260721, 20261721]),
            ("seeds", (20261721, 20260721)),
            ("folds", True),
            ("normal_rows", 69.0),
            ("source_width", [96]),
            ("interpolated_width", 139),
            ("onset", True),
            ("warp_factor", 1),
            ("early_start", False),
            ("late_start", 83),
            ("window_length", 55),
            ("acf_lag_min", 1.0),
            ("acf_lag_max", 26),
            ("peak_lag_min", 1),
            ("peak_lag_max", 24.0),
            ("method_order", ["C0", "E12", "C12"]),
            ("method_order", ("C0", "C12", "E12")),
            ("change_types", tuple(reversed(config.change_types))),
            ("required_speedup", 1),
        )
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
