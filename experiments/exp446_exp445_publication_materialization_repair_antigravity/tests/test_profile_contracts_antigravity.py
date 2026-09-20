"""Closed Exp446 profile, checkpoint, readiness, runtime, and reporting contracts."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import unittest
from unittest import mock


_COMMIT_ANTIGRAVITY = "a" * 40


class Exp446ProfileContractsAntigravityTests(unittest.TestCase):
    """Exp446 may be reached only through its reviewed singleton identity."""

    def _profile_antigravity(self):
        from experiments.exp444_train_only_stability_precision_gate_antigravity import profiles_antigravity as profiles

        self.assertTrue(hasattr(profiles, "EXP446_PROFILE_ANTIGRAVITY"))
        return profiles.EXP446_PROFILE_ANTIGRAVITY

    def test_exp446_singleton_rejects_equal_minted_and_altered_profiles_antigravity(self) -> None:
        """Identity comparison prevents caller-selected roots or schemas."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import (
            profiles_antigravity as profiles,
            readiness_antigravity as readiness,
            reporting_antigravity as reporting,
            runtime_antigravity as runtime,
        )

        profile = self._profile_antigravity()
        self.assertEqual(
            (profile.experiment_id, profile.schema, profile.run_identity, profile.run_root),
            (
                "Exp446",
                "exp446",
                "exp446-exp445-publication-materialization-repair-antigravity-20260811",
                Path("/Users/mac/Documents/Dataset/exp446_exp445_publication_materialization_repair_antigravity_20260811"),
            ),
        )
        minted = replace(profile)
        self.assertEqual(minted, profile)
        for route in (
            readiness._require_closed_profile_antigravity,
            reporting._require_publication_profile_antigravity,
        ):
            with self.subTest(route=route.__name__), self.assertRaisesRegex(ValueError, "profile"):
                route(minted)
        with self.assertRaisesRegex(ValueError, "profile"):
            runtime.run_new_for_profile_antigravity(minted, _COMMIT_ANTIGRAVITY)
        for changed in (
            {"run_root": Path("/tmp/exp446_changed_antigravity")},
            {"schema": "exp445"},
            {"experiment_id": "Exp445"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(profile, **changed)

    def test_exp446_checkpoint_context_binds_its_schema_documents_and_label_free_payload_antigravity(self) -> None:
        """Exp445 context or a label-capable payload cannot be reused for Exp446."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
            ProfileCheckpointContextAntigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        context = runtime._context_for_profile_antigravity(
            self._profile_antigravity(), _COMMIT_ANTIGRAVITY,
        )
        self.assertEqual(
            (context.experiment_id, context.schema, context.outer_workers, context.inner_threads),
            ("Exp446", "exp446", 11, 1),
        )
        payload = context.to_payload()
        payload["label_access_count"] = 0
        with self.assertRaises(ValueError):
            ProfileCheckpointContextAntigravity.from_payload(payload)
        exp445_context = ProfileCheckpointContextAntigravity(
            "Exp445", "exp445", "exp445-exp444-publication-source-authority-repair-antigravity-20260810",
            context.scientific_config_sha256, context.environment_lock_sha256,
            30, 11, 1, _COMMIT_ANTIGRAVITY,
            "099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d",
            "20d73ed9cebfbd4cd40d4d3354197907483ac9092a8392e93e9bf1b44027a37a",
        )
        validator = getattr(
            ProfileCheckpointContextAntigravity,
            "from_exp446_payload_antigravity",
            None,
        )
        self.assertTrue(callable(validator))
        with self.assertRaisesRegex(ValueError, "^Exp446 checkpoint context is invalid$"):
            validator(exp445_context.to_payload())
        self.assertEqual(validator(context.to_payload()), context)

    def test_exp446_runtime_rejects_every_override_before_live_gates_antigravity(self) -> None:
        """Fixed-profile routing must run before predecessor, source, or root work."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        profile = self._profile_antigravity()
        reviewed_commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), text=True,
        ).strip()
        expected = "Exp446 supports fixed new execution only"
        for mode, root, attempt, executor in (
            ("resume", None, None, None),
            ("new", Path("/tmp/exp446_override_antigravity"), None, None),
            ("new", None, "attempt_0001_antigravity", None),
            ("new", None, None, object()),
        ):
            with self.subTest(mode=mode, root=root, attempt=attempt, executor=executor is not None):
                guards = {
                    "review": mock.Mock(side_effect=AssertionError("review gate reached")),
                    "predecessor": mock.Mock(side_effect=AssertionError("predecessor reached")),
                    "source": mock.Mock(side_effect=AssertionError("source reached")),
                    "root": mock.Mock(side_effect=AssertionError("root reached")),
                }
                with (
                    mock.patch.object(runtime, "_validate_reviewed_commit_antigravity", guards["review"]),
                    mock.patch.object(runtime, "_verify_exp446_predecessor_antigravity", guards["predecessor"]),
                    mock.patch.object(runtime, "_load_source_inputs_antigravity", guards["source"]),
                    mock.patch.object(runtime, "_attempt_root_antigravity", guards["root"]),
                ):
                    try:
                        runtime._execute_runtime_antigravity(
                            reviewed_commit=reviewed_commit,
                            mode=mode,
                            attempt_name=attempt,
                            run_root=root,
                            executor_factory=executor,
                            profile=profile,
                        )
                    except ValueError as error:
                        actual = str(error)
                    except AssertionError as error:
                        actual = str(error)
                self.assertEqual(actual, expected)
                self.assertEqual({name: guard.call_count for name, guard in guards.items()}, {
                    "review": 0, "predecessor": 0, "source": 0, "root": 0,
                })


if __name__ == "__main__":
    unittest.main()
