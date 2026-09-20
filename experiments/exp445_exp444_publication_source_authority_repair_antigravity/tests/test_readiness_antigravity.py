"""Exp445 fixed-profile readiness behavior, before its lifecycle exists."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


class Exp445ReadinessAntigravityTests(unittest.TestCase):
    def _profile_antigravity(self, root: Path):
        from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP445_PROFILE_ANTIGRAVITY

        profile = object.__new__(type(EXP445_PROFILE_ANTIGRAVITY))
        for name in (
            "experiment_id", "schema", "run_identity", "reviewed_design_path",
            "preregistration_path", "predecessor_required",
        ):
            object.__setattr__(profile, name, getattr(EXP445_PROFILE_ANTIGRAVITY, name))
        object.__setattr__(profile, "run_root", root)
        return profile

    def test_profile_readiness_boundary_matrix_preserves_official_root_state_antigravity(self) -> None:
        """Every gate leaves both the test-local and immutable official roots untouched."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        commit = "a" * 40
        with tempfile.TemporaryDirectory(prefix="exp445_readiness_matrix_antigravity_") as directory:
            official_root = readiness.EXP445_PROFILE_ANTIGRAVITY.run_root
            official_present = official_root.exists() or official_root.is_symlink()
            official_identity = official_root.lstat() if official_present else None
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
            profile = self._profile_antigravity(root)
            good_environment = SimpleNamespace(inner_threads=1, outer_workers=11)
            good_provenance = SimpleNamespace(backend_decision=SimpleNamespace(backend="cpu", workers=11))
            good_controls = tuple(SimpleNamespace(matches_source=True) for _ in range(30))
            good_documents = {
                "reviewed_design_sha256": "099c0d412d2375764a7e30440feca17d70a2a6bbfdd14e3e53999fe3c458d70d",
                "preregistration_sha256": "20d73ed9cebfbd4cd40d4d3354197907483ac9092a8392e93e9bf1b44027a37a",
            }
            frozen_config = readiness.FROZEN_CONFIG
            predecessor_path = (
                "experiments.exp445_exp444_publication_source_authority_repair_antigravity."
                "predecessor_antigravity.verify_exp444_predecessor_antigravity"
            )

            def run_case(failure: str | None = None) -> tuple[object | None, tuple[str, ...]]:
                """Use the real closed Exp445 entrypoint; fake only each external gate's evidence."""
                calls: list[str] = []

                def gate(name: str, value: object = None):
                    def invoke(*_args, **_kwargs):
                        calls.append(name)
                        if failure == name and name != "labels":
                            raise ValueError(f"{name} fault")
                        return value
                    return invoke

                class ConfigGateAntigravity:
                    def __getattr__(self, name: str) -> object:
                        return getattr(frozen_config, name)

                    def validate(self) -> None:
                        calls.append("scientific_config")
                        if failure == "scientific_config":
                            raise ValueError("scientific config fault")
                        frozen_config.validate()

                materialized = (tuple(range(30)), {"database": "b" * 64}, 1 if failure == "labels" else 0)
                def current_head() -> str:
                    calls.append("current_commit")
                    return "c" * 40 if failure == "wrong_commit" else commit
                with (
                    mock.patch.object(readiness, "EXP445_PROFILE_ANTIGRAVITY", profile),
                    mock.patch.object(readiness, "current_clean_head_antigravity", side_effect=gate("dirty_commit") if failure == "dirty_commit" else current_head),
                    mock.patch.object(readiness, "load_worker_parity_authority_antigravity", side_effect=gate("worker_parity")),
                    mock.patch.object(readiness, "_profile_document_sha256_antigravity", side_effect=gate("documents", good_documents)),
                    mock.patch.object(readiness, "FROZEN_CONFIG", ConfigGateAntigravity()),
                    mock.patch.object(readiness, "verify_environment_antigravity", side_effect=gate("environment_package_topology", SimpleNamespace(inner_threads=1, outer_workers=10) if failure == "environment_package_topology" else good_environment)),
                    mock.patch.object(readiness, "verify_exp444_provenance_antigravity", side_effect=gate("provenance_backend", SimpleNamespace(backend_decision=SimpleNamespace(backend="mlx", workers=11)) if failure == "provenance_backend" else good_provenance)),
                    mock.patch(predecessor_path, side_effect=gate("predecessor")),
                    mock.patch.object(readiness, "_materialize_values_antigravity", side_effect=gate("labels", materialized)),
                    mock.patch.object(readiness, "replay_original_candidates_antigravity", side_effect=gate("controls", good_controls)),
                ):
                    if failure in {"wrong_commit", "environment_package_topology", "provenance_backend", "labels"}:
                        with self.assertRaises(ValueError):
                            readiness.readiness_for_profile_antigravity(profile, commit)
                        return None, tuple(calls)
                    try:
                        return readiness.readiness_for_profile_antigravity(profile, commit), tuple(calls)
                    except ValueError:
                        if failure is None:
                            raise
                        return None, tuple(calls)

            evidence, calls = run_case()
            self.assertIsNotNone(evidence)
            self.assertEqual(calls, (
                "current_commit", "worker_parity", "documents", "scientific_config",
                "environment_package_topology", "provenance_backend", "predecessor",
                "labels", "controls", "documents",
            ))
            self.assertEqual((evidence.to_payload_antigravity()["schema"], evidence.workers, evidence.label_access_count), ("exp445.readiness.v1", 11, 0))
            self.assertFalse(root.exists())

            cases = {
                "wrong_commit": ("current_commit",),
                "dirty_commit": ("dirty_commit",),
                "worker_parity": ("current_commit", "worker_parity"),
                "documents": ("current_commit", "worker_parity", "documents"),
                "scientific_config": ("current_commit", "worker_parity", "documents", "scientific_config"),
                "environment_package_topology": ("current_commit", "worker_parity", "documents", "scientific_config", "environment_package_topology"),
                "provenance_backend": ("current_commit", "worker_parity", "documents", "scientific_config", "environment_package_topology", "provenance_backend"),
                "predecessor": ("current_commit", "worker_parity", "documents", "scientific_config", "environment_package_topology", "provenance_backend", "predecessor"),
                "labels": ("current_commit", "worker_parity", "documents", "scientific_config", "environment_package_topology", "provenance_backend", "predecessor", "labels", "controls", "documents"),
            }
            for name, expected_calls in cases.items():
                with self.subTest(name=name):
                    _evidence, observed_calls = run_case(name)
                    self.assertEqual(observed_calls, expected_calls)
                self.assertFalse(root.exists(), name)

            for name, make_root in (("existing_root", lambda: root.mkdir()), ("symlink_root", lambda: root.symlink_to(Path(directory)))):
                with self.subTest(name=name):
                    make_root()
                    identity_before = root.lstat()
                    target_before = root.resolve() if root.is_symlink() else None
                    with mock.patch.object(readiness, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(readiness, "current_clean_head_antigravity", return_value=commit), mock.patch.object(readiness, "load_worker_parity_authority_antigravity"), mock.patch.object(readiness, "_profile_document_sha256_antigravity", return_value=good_documents):
                        with self.assertRaises(ValueError):
                            readiness.readiness_for_profile_antigravity(profile, commit)
                    self.assertTrue(root.exists() or root.is_symlink())
                    self.assertEqual((root.lstat().st_dev, root.lstat().st_ino), (identity_before.st_dev, identity_before.st_ino))
                    if target_before is not None:
                        self.assertEqual(root.resolve(), target_before)
                    root.unlink() if root.is_symlink() else root.rmdir()
                self.assertFalse(root.exists(), name)
            self.assertEqual(official_root.exists() or official_root.is_symlink(), official_present)
            if official_identity is not None:
                self.assertEqual(
                    (official_root.lstat().st_dev, official_root.lstat().st_ino),
                    (official_identity.st_dev, official_identity.st_ino),
                )
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
        public_names = {name for name in vars(lifecycle_antigravity) if not name.startswith("_")}
        self.assertEqual(
            public_names & {
                "EXP445_PROFILE_ANTIGRAVITY", "readiness_for_profile_antigravity",
                "run_new_for_profile_antigravity",
            },
            set(),
        )

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
