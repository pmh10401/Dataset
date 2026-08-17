from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_binds_exp456_identity_and_exp455_scientific_gates_antigravity(self) -> None:
        """Break caught: Exp456 identity or a frozen scientific gate drifts before evaluation."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.config_antigravity import FROZEN_CONFIG_ANTIGRAVITY

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp456")
        self.assertEqual(config.kind, "exp455_dataset_key_order_repair")
        self.assertEqual(config.run_identity, "exp456-exp455-dataset-key-order-repair-antigravity-20260817")
        self.assertEqual(config.run_root, "/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817")
        self.assertEqual(config.package, "experiments/exp456_exp455_dataset_key_order_repair_antigravity")
        self.assertEqual(config.approved_design_sha256, "79c6ad3e8bfad6f7f9b397fc1d20b448ebc656f198de5d449ff897a92de98247")
        self.assertEqual(config.candidates, ("ECG5000:hydra_local_gap", "ChlorineConcentration:multirocket_local_gap"))
        self.assertEqual(config.target_family_counts, (("ECG5000", 5), ("ChlorineConcentration", 3)))
        self.assertEqual((config.global_fn_must_be_below, config.global_fp_max, config.global_f1_min), (1988, 314, 0.6264199935))
        with self.assertRaises(FrozenInstanceError):
            config.experiment = "changed"  # type: ignore[misc]
