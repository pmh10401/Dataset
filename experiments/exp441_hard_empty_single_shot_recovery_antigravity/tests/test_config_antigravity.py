"""Frozen configuration behavior for Exp441."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import unittest

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    Exp441FrozenConfig,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    governance_payload_antigravity,
)


class Exp441ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_identity_and_cpu_policy(self) -> None:
        """Changing a frozen experiment literal must be rejected."""
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp441")
        self.assertEqual(FROZEN_CONFIG.reviewed_design_sha256, "c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da")
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_CONFIG.candidate_budget, 30)
        self.assertEqual(FROZEN_CONFIG.candidate_per_dataset, 1)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 44120260808)
        self.assertEqual(FROZEN_CONFIG.bootstrap_samples, 20000)
        self.assertEqual(FROZEN_CONFIG.numpy_version, "2.4.6")
        self.assertEqual(cpu_worker_count_antigravity(12), 11)
        self.assertFalse(FROZEN_CONFIG.tensor_eligible)
        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG.dataset_count = 1  # type: ignore[misc]

    def test_governance_is_exact_flat_and_strictly_typed(self) -> None:
        """A truthy or nested governance replacement must not satisfy the contract."""
        governance = governance_payload_antigravity()
        self.assertEqual(
            set(governance),
            {
                "official_run_purpose", "bootstrap_interpretation", "adaptive_followup",
                "independent_validation", "external_generalization_supported",
                "confirmatory_statistical_claim", "scientific_winner_selected",
                "review_only", "operational_alert_eligible",
            },
        )
        self.assertEqual(governance["official_run_purpose"], "deterministic_retrospective_reproduction")
        self.assertEqual(governance["bootstrap_interpretation"], "descriptive_post_selection")
        self.assertFalse(FROZEN_CONFIG.external_generalization_supported)
        self.assertFalse(FROZEN_CONFIG.scientific_winner_selected)
        for key in set(governance) - {"official_run_purpose", "bootstrap_interpretation"}:
            self.assertIs(type(governance[key]), bool, key)
        self.assertTrue(all(not isinstance(value, (dict, list, tuple)) for value in governance.values()))

    def test_serialized_config_excludes_forbidden_evaluation_sources(self) -> None:
        """Adding either barred Exp440 evaluation source to config would leak prediction access."""
        serialized = json.dumps(frozen_config_payload_antigravity(), sort_keys=True)
        self.assertNotIn(
            "/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808/evaluation_antigravity",
            serialized,
        )
        self.assertNotIn(
            "experiments/exp440_train_only_low_frequency_precision_gate_antigravity/evaluation_antigravity.py",
            serialized,
        )

    def test_cpu_policy_rejects_non_positive_or_non_integer_counts(self) -> None:
        """An invalid host CPU value must not silently produce a worker topology."""
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    cpu_worker_count_antigravity(value)  # type: ignore[arg-type]

    def test_constructor_rejects_equal_but_wrong_literal_types(self) -> None:
        """Numeric or truthy aliases must not create an apparently frozen override surface."""
        for field, value in (("candidate_budget", 30.0), ("review_only", 1)):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Exp441FrozenConfig(**{field: value})  # type: ignore[arg-type]
