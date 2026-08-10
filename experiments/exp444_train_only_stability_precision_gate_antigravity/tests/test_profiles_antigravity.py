"""Contract tests for the closed Exp444/Exp445 lifecycle identities."""
from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity import cli_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import (
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
    scientific_config_sha256_antigravity,
    validate_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CheckpointContextAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import (
    EXP444_PROFILE_ANTIGRAVITY,
    EXP445_PROFILE_ANTIGRAVITY,
    LifecycleProfileAntigravity,
)


_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]


class LifecycleProfilesAntigravityTests(unittest.TestCase):
    """A changed identity, root, or schema must not mint a lifecycle profile."""

    def test_only_two_fixed_profiles_separate_identity_and_root_antigravity(self) -> None:
        profiles = (EXP444_PROFILE_ANTIGRAVITY, EXP445_PROFILE_ANTIGRAVITY)
        exported = tuple(
            value for name, value in vars(__import__(
                "experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity",
                fromlist=["*"],
            )).items()
            if name.endswith("_PROFILE_ANTIGRAVITY")
        )
        self.assertEqual(exported, profiles)
        self.assertEqual(EXP444_PROFILE_ANTIGRAVITY.experiment_id, "Exp444")
        self.assertEqual(EXP444_PROFILE_ANTIGRAVITY.schema, "exp444")
        self.assertEqual(
            EXP444_PROFILE_ANTIGRAVITY.run_identity,
            "exp444-train-only-stability-precision-gate-antigravity-20260810",
        )
        self.assertEqual(
            EXP444_PROFILE_ANTIGRAVITY.run_root,
            Path("/Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810"),
        )
        self.assertEqual(
            EXP444_PROFILE_ANTIGRAVITY.reviewed_design_path,
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md",
        )
        self.assertEqual(
            EXP444_PROFILE_ANTIGRAVITY.preregistration_path,
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP444_PREREGISTRATION_antigravity_KO.md",
        )
        self.assertFalse(EXP444_PROFILE_ANTIGRAVITY.predecessor_required)
        self.assertEqual(EXP445_PROFILE_ANTIGRAVITY.experiment_id, "Exp445")
        self.assertEqual(EXP445_PROFILE_ANTIGRAVITY.schema, "exp445")
        self.assertEqual(
            EXP445_PROFILE_ANTIGRAVITY.run_identity,
            "exp445-exp444-publication-source-authority-repair-antigravity-20260810",
        )
        self.assertEqual(
            EXP445_PROFILE_ANTIGRAVITY.run_root,
            Path("/Users/mac/Documents/Dataset/exp445_exp444_publication_source_authority_repair_antigravity_20260810"),
        )
        self.assertEqual(
            EXP445_PROFILE_ANTIGRAVITY.reviewed_design_path,
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/superpowers/specs/2026-08-10-exp445-exp444-publication-source-authority-repair-antigravity-design.md",
        )
        self.assertEqual(
            EXP445_PROFILE_ANTIGRAVITY.preregistration_path,
            _REPOSITORY_ROOT_ANTIGRAVITY / "docs/EXP445_PREREGISTRATION_antigravity_KO.md",
        )
        self.assertTrue(EXP445_PROFILE_ANTIGRAVITY.predecessor_required)
        self.assertNotEqual(EXP444_PROFILE_ANTIGRAVITY.run_root, EXP445_PROFILE_ANTIGRAVITY.run_root)

    def test_profile_rejects_arbitrary_identity_root_and_schema_antigravity(self) -> None:
        for changes in (
            {"experiment_id": "Exp446"},
            {"run_root": Path("/tmp/changed_antigravity")},
            {"schema": "exp446"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(EXP445_PROFILE_ANTIGRAVITY, **changes)

        with self.assertRaises(ValueError):
            LifecycleProfileAntigravity(
                experiment_id="Exp446",
                schema="exp446",
                run_identity="arbitrary-antigravity",
                run_root=Path("/tmp/arbitrary_antigravity"),
                reviewed_design_path=Path("/tmp/design_antigravity.md"),
                preregistration_path=Path("/tmp/preregistration_antigravity.md"),
                predecessor_required=False,
            )

    def test_exp444_public_signatures_remain_frozen_antigravity(self) -> None:
        expected = {
            "cpu_worker_count_antigravity": "(logical_cpus: 'int | None' = None) -> 'int'",
            "frozen_config_payload_antigravity": "() -> 'dict[str, object]'",
            "scientific_config_sha256_antigravity": "() -> 'str'",
            "validate_antigravity": "(config: 'Exp444FrozenConfig | None' = None) -> 'bool'",
            "CheckpointContextAntigravity": "(experiment_id: 'str', run_identity: 'str', scientific_config_sha256: 'str', environment_lock_sha256: 'str', total_units: 'int', outer_workers: 'int', inner_threads: 'int', reviewed_commit: 'str') -> None",
            "build_parser_antigravity": "() -> 'argparse.ArgumentParser'",
            "main_antigravity": "(argv: 'tuple[str, ...] | None' = None) -> 'int'",
        }
        actual = {
            "cpu_worker_count_antigravity": str(inspect.signature(cpu_worker_count_antigravity)),
            "frozen_config_payload_antigravity": str(inspect.signature(frozen_config_payload_antigravity)),
            "scientific_config_sha256_antigravity": str(inspect.signature(scientific_config_sha256_antigravity)),
            "validate_antigravity": str(inspect.signature(validate_antigravity)),
            "CheckpointContextAntigravity": str(inspect.signature(CheckpointContextAntigravity)),
            "build_parser_antigravity": str(inspect.signature(cli_antigravity.build_parser_antigravity)),
            "main_antigravity": str(inspect.signature(cli_antigravity.main_antigravity)),
        }
        self.assertEqual(actual, expected)
