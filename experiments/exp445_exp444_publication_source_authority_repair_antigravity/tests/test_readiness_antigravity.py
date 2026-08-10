"""Exp445 fixed-profile readiness behavior, before its lifecycle exists."""
from __future__ import annotations

import inspect
import unittest


class Exp445ReadinessAntigravityTests(unittest.TestCase):
    def test_exp445_readiness_evidence_keeps_its_own_document_authority_antigravity(self) -> None:
        """Substituting Exp444 document hashes would hide an Exp445 governance drift."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity import ReadinessEvidenceAntigravity

        evidence = ReadinessEvidenceAntigravity(
            "a" * 40, {"database": "b" * 64}, 30, 30, 0, "cpu", 11,
            "8835702b9793a5cc08b5e95621b49c0a4a4136325354da7a655761f6b91f3a94",
            "8ba0a9524600bef7479df0626834bfeef4b3410f64d3f2d1cbd3885c784eb11a",
            "099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d",
            "20d73ed9cebfbd4cd40d4d3354197907483ac9092a8392e93e9bf1b44027a37a",
            "79b6a5847a01f594d140c73aa587589413b4933ac7d53dcf07d2bb478a5f771e", "Exp445",
        )
        self.assertEqual(evidence.to_payload_antigravity()["schema"], "exp445.readiness.v1")

    def test_fixed_readiness_exposes_only_reviewed_commit_antigravity(self) -> None:
        """Adding caller-controlled routing would let a new identity evade the frozen profile."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity

        signature = inspect.signature(lifecycle_antigravity.run_readiness_antigravity)
        self.assertEqual(tuple(signature.parameters), ("reviewed_commit",))

        execution_signature = inspect.signature(lifecycle_antigravity.run_new_antigravity)
        self.assertEqual(tuple(execution_signature.parameters), ("reviewed_commit",))
        self.assertFalse(hasattr(lifecycle_antigravity, "run_resume_antigravity"))

    def test_profile_aware_readiness_rejects_a_caller_minted_profile_antigravity(self) -> None:
        """Accepting an equal-looking profile would make root and document authority selectable."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import (
            LifecycleProfileAntigravity,
        )

        forged = object.__new__(LifecycleProfileAntigravity)
        object.__setattr__(forged, "experiment_id", "Exp445")
        object.__setattr__(forged, "schema", "exp445")
        object.__setattr__(forged, "run_identity", "exp445-exp444-publication-source-authority-repair-antigravity-20260810")
        object.__setattr__(forged, "run_root", None)
        object.__setattr__(forged, "reviewed_design_path", None)
        object.__setattr__(forged, "preregistration_path", None)
        object.__setattr__(forged, "predecessor_required", True)
        with self.assertRaisesRegex(ValueError, "profile"):
            readiness_antigravity.readiness_for_profile_antigravity(forged, "a" * 40)


if __name__ == "__main__":
    unittest.main()
