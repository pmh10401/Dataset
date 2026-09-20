from dataclasses import asdict
from pathlib import Path
import unittest

from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP436_PROFILE,
    EXP437_PROFILE,
    ExecutionProfileAntigravity,
    frozen_config_for_profile_antigravity,
    validated_profile_antigravity,
)


class ExecutionProfileAntigravityTests(unittest.TestCase):
    def test_exp437_identity_and_fixed_sibling_roots_are_frozen(self):
        self.assertEqual(EXP437_PROFILE.experiment_id, "Exp437")
        self.assertEqual(
            EXP437_PROFILE.run_identity,
            "exp437-metadata-roundtrip-antigravity-20260807",
        )
        self.assertEqual(
            EXP437_PROFILE.run_parent,
            Path("/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807"),
        )
        self.assertEqual(EXP437_PROFILE.output, EXP437_PROFILE.run_parent / "output_antigravity")
        self.assertEqual(EXP437_PROFILE.checkpoint, EXP437_PROFILE.run_parent / "checkpoint_antigravity")
        self.assertEqual(EXP437_PROFILE.readiness, EXP437_PROFILE.run_parent / "readiness_antigravity.json")
        self.assertEqual(EXP437_PROFILE.evaluation, EXP437_PROFILE.run_parent / "evaluation_antigravity")

    def test_profiles_differ_only_in_execution_identity_fields(self):
        left = asdict(frozen_config_for_profile_antigravity(EXP436_PROFILE))
        right = asdict(frozen_config_for_profile_antigravity(EXP437_PROFILE))
        self.assertEqual(
            {key for key in left if left[key] != right[key]},
            {"experiment_id", "run_identity"},
        )

    def test_arbitrary_profile_is_rejected(self):
        forged = ExecutionProfileAntigravity(
            experiment_id="Exp999",
            run_identity="forged-antigravity",
            cli_program="forged_antigravity.py",
            run_parent=Path("/tmp/forged_antigravity"),
        )
        with self.assertRaisesRegex(ValueError, "frozen execution profile"):
            validated_profile_antigravity(forged)
