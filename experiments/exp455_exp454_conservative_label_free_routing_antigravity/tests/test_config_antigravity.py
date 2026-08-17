from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import unittest

from experiments.exp455_exp454_conservative_label_free_routing_antigravity.config_antigravity import (
    Exp455ConfigAntigravity,
    FROZEN_CONFIG_ANTIGRAVITY,
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_binds_the_approved_exp455_identity_antigravity(self) -> None:
        """Break caught: a future phase can silently change the approved study identity."""
        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertIsInstance(config, Exp455ConfigAntigravity)
        self.assertEqual(config.experiment, "Exp455")
        self.assertEqual(config.kind, "conservative_label_free_routing")
        self.assertEqual(config.run_identity, "exp455-exp454-conservative-label-free-routing-antigravity-20260817")
        self.assertEqual(
            config.run_root,
            "/Users/mac/Documents/Dataset/exp455_exp454_conservative_label_free_routing_antigravity_20260817",
        )
        self.assertEqual(config.package, "experiments/exp455_exp454_conservative_label_free_routing_antigravity")
        self.assertEqual(
            config.approved_design_path,
            "docs/superpowers/specs/2026-08-17-exp455-conservative-label-free-routing-antigravity-design.html",
        )
        self.assertEqual(config.approved_design_sha256, "ae391bdaa9b34b8d603722262887143e0bdf8284bdcbf0fa1b059d4841c0369a")

    def test_frozen_config_binds_candidates_population_gates_and_immutable_evidence_antigravity(self) -> None:
        """Break caught: routing scope or its immutable Exp454 evidence drifts before evaluation."""
        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(
            config.candidates,
            ("ECG5000:hydra_local_gap", "ChlorineConcentration:multirocket_local_gap"),
        )
        self.assertEqual(config.target_family_counts, (("ECG5000", 5), ("ChlorineConcentration", 3)))
        self.assertEqual(config.target_dataset_count, 8)
        self.assertEqual(config.baseline_dataset_count, 40)
        self.assertEqual(config.global_fn_must_be_below, 1988)
        self.assertEqual(config.global_fp_max, 314)
        self.assertEqual(config.global_f1_min, 0.6264199935)
        self.assertEqual(config.target_family_fn_delta_max, 0)
        self.assertEqual(
            config.exp454_sha256,
            (
                ("artifact_map", "748fdc62de555279640b4ce588b3155670ebe33e13e5997d40061d96011bff65"),
                ("evaluation", "c2b6321d18ba4c98e426be96ea3d3fb0824e75926fa54ba4600634dded68e1ac"),
                ("decision", "3510a426d3ee083ccb3f8599c64d0c91cc56d0b902780c8b6a2136098cd9a595"),
            ),
        )
        self.assertEqual(
            config.official_python,
            "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python",
        )
        self.assertTrue(config.router_inputs_label_free)
        self.assertFalse(config.mlx_assessment_required)

    def test_frozen_config_is_a_slots_frozen_dataclass_antigravity(self) -> None:
        """Break caught: later code can mutate or attach unregistered routing configuration."""
        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertGreater(len(fields(Exp455ConfigAntigravity)), 0)
        with self.assertRaises(FrozenInstanceError):
            config.experiment = "changed"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            config.unregistered = "changed"  # type: ignore[attr-defined]
