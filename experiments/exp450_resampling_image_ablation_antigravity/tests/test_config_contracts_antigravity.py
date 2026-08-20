"""Frozen identity and public-contract tests for Exp450."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    FROZEN_EXP450_CONFIG,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
)


class Exp450ConfigTests(unittest.TestCase):
    def test_four_conditions_are_exact_factorial(self) -> None:
        """A swapped mode or condition ordering must reject the approved ablation."""
        self.assertEqual(
            tuple((condition.condition_id, condition.length_mode, condition.image_mode)
                  for condition in FROZEN_EXP450_CONFIG.conditions),
            (
                ("A", "linear", "linear"),
                ("B", "polyphase_pchip", "linear"),
                ("C", "linear", "area"),
                ("D", "polyphase_pchip", "area"),
            ),
        )

    def test_population_statistics_and_roots_are_frozen(self) -> None:
        """A population or runtime-namespace drift must fail before execution."""
        self.assertEqual(FROZEN_EXP450_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP450_CONFIG.test_row_count, 195900)
        self.assertEqual(len(FROZEN_EXP450_CONFIG.smoke_datasets), 23)
        self.assertTrue(all("antigravity" in root.name for root in FROZEN_EXP450_CONFIG.roots))

    def test_condition_rejects_a_nonapproved_factor_pair(self) -> None:
        """Condition D with a linear image must never represent the ablation."""
        with self.assertRaisesRegex(ValueError, "approved factorial"):
            Exp450Condition("D", "polyphase_pchip", "linear")

    def test_backend_record_preserves_cpu_worker_provenance(self) -> None:
        """A backend decision must expose its frozen execution provenance."""
        decision = Exp450BackendDecision("mps", "2.13.0", True, False, 12, 11)
        self.assertEqual(
            decision.artifact_record(),
            {
                "tensor_backend": "mps",
                "torch_version": "2.13.0",
                "mps_available": True,
                "mlx_available": False,
                "logical_cpu_count": 12,
                "cpu_workers": 11,
                "inner_numerical_threads": 1,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            decision.cpu_workers = 1  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
