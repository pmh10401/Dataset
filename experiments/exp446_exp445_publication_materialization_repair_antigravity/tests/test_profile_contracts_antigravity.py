"""Closed Exp446 profile, checkpoint, readiness, runtime, and reporting contracts."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest


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
        exp445_payload = context.to_payload()
        exp445_payload["experiment_id"] = "Exp445"
        exp445_payload["schema"] = "exp445"
        with self.assertRaises(ValueError):
            ProfileCheckpointContextAntigravity.from_payload(exp445_payload)

    def test_exp446_runtime_rejects_nonfixed_mode_root_and_attempt_before_mutation_antigravity(self) -> None:
        """Routing overrides cannot select resume or a caller-owned filesystem target."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        profile = self._profile_antigravity()
        for mode, root, attempt in (
            ("resume", None, None),
            ("new", Path("/tmp/exp446_override_antigravity"), None),
            ("new", None, "attempt_0001_antigravity"),
        ):
            with self.subTest(mode=mode, root=root, attempt=attempt), self.assertRaises(ValueError):
                runtime._execute_runtime_antigravity(
                    reviewed_commit=_COMMIT_ANTIGRAVITY,
                    mode=mode,
                    attempt_name=attempt,
                    run_root=root,
                    executor_factory=None,
                    profile=profile,
                )


if __name__ == "__main__":
    unittest.main()
