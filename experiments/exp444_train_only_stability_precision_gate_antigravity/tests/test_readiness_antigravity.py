"""Behavior tests for Exp444's immutable pre-run readiness boundary."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class ReadinessAntigravityTests(unittest.TestCase):
    def test_reviewed_design_bytes_are_read_from_the_actual_pinned_path_antigravity(self) -> None:
        """Changed design bytes must block readiness."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        self.assertEqual(readiness._review_document_sha256_antigravity(), {
            "reviewed_design_sha256": "1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8",
            "preregistration_sha256": "347ba8f0e37d2f185b98ed65acdc936fd2238cea5e97ed0fecd34cabcc1974db",
        })
        with tempfile.TemporaryDirectory(prefix="exp444_document_drift_antigravity_") as directory:
            changed = Path(directory) / "design_antigravity.md"
            changed.write_text("changed design\n")
            with patch.object(readiness, "_REVIEWED_DESIGN_PATH_ANTIGRAVITY", changed):
                with self.assertRaisesRegex(ValueError, "design"):
                    readiness._review_document_sha256_antigravity()

    def test_preregistration_bytes_are_read_from_the_actual_pinned_path_antigravity(self) -> None:
        """Changed preregistration bytes must block readiness independently of the design."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        with tempfile.TemporaryDirectory(prefix="exp444_preregistration_drift_antigravity_") as directory:
            changed = Path(directory) / "preregistration_antigravity.md"
            changed.write_text("changed preregistration\n")
            with patch.object(readiness, "_PREREGISTRATION_PATH_ANTIGRAVITY", changed):
                with self.assertRaisesRegex(ValueError, "preregistration"):
                    readiness._review_document_sha256_antigravity()

    def test_worker_parity_authority_rejects_byte_and_payload_drift_antigravity(self) -> None:
        """Changed Task7 bytes or a false parity value cannot become readiness evidence."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        authority = readiness.load_worker_parity_authority_antigravity()
        self.assertEqual(authority["candidate_count"], 30)
        self.assertTrue(authority["canonical_byte_equal"])
        self.assertEqual((authority["workers"], authority["label_access_count"]), (11, 0))
        self.assertEqual(
            (authority["sequential_sha256"], authority["parallel_sha256"]),
            ("729b31c0698fda6b60c17012a3fd2cae3f4b0899e01e5a3bf4d569902e814857",) * 2,
        )
        changed = dict(authority)
        changed["canonical_byte_equal"] = False
        with self.assertRaisesRegex(ValueError, "parity"):
            readiness.validate_worker_parity_authority_payload_antigravity(changed)
        with tempfile.TemporaryDirectory(prefix="exp444_parity_bytes_antigravity_") as directory:
            authority_path = Path(directory) / "worker_parity_authority_antigravity.json"
            authority_path.write_bytes(json.dumps(authority, sort_keys=True).encode() + b"\n")
            with patch.object(readiness, "_WORKER_PARITY_AUTHORITY_PATH_ANTIGRAVITY", authority_path):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    readiness.load_worker_parity_authority_antigravity()

    def test_public_readiness_interface_binds_the_reviewed_commit_antigravity(self) -> None:
        """A missing readiness boundary would allow an unreviewed lifecycle launch."""
        module = importlib.import_module(
            "experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity"
        )
        self.assertTrue(callable(module.run_readiness_antigravity))
        self.assertTrue(callable(module.validate_readiness_for_run_antigravity))

    def test_source_failure_preserves_the_absent_fixed_root_antigravity(self) -> None:
        """Creating the fixed root before source verification would leave failed ownership evidence."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        with tempfile.TemporaryDirectory(prefix="exp444_readiness_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            with (
                patch.object(readiness, "_run_root_antigravity", return_value=root),
                patch.object(readiness, "current_clean_head_antigravity", return_value="a" * 40),
                patch.object(readiness, "verify_environment_antigravity", return_value=types.SimpleNamespace(inner_threads=1, outer_workers=11)),
                patch.object(readiness, "verify_exp444_provenance_antigravity", side_effect=ValueError("source changed")),
            ):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    readiness.run_readiness_antigravity("a" * 40)
            self.assertFalse(root.exists())

    def test_readiness_payload_rejects_changed_source_or_label_governance_antigravity(self) -> None:
        """A relaxed payload replay could bind source drift or TEST-label access to a new run."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import readiness_antigravity as readiness

        payload = readiness._readiness_payload_antigravity(
            "a" * 40,
            source_hashes={"database": readiness.FROZEN_CONFIG.database_sha256},
            source_candidate_count=30,
            control_count=30,
            label_access_count=0,
        )
        self.assertEqual(readiness.validate_readiness_payload_antigravity(payload)["reviewed_commit"], "a" * 40)
        self.assertEqual(payload["reviewed_design_sha256"], "1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8")
        self.assertEqual(payload["preregistration_sha256"], "347ba8f0e37d2f185b98ed65acdc936fd2238cea5e97ed0fecd34cabcc1974db")
        self.assertEqual(payload["worker_parity_authority_sha256"], readiness.FROZEN_CONFIG.worker_parity_authority_sha256)
        for key, value in (("label_access_count", 1), ("source_candidate_count", 29)):
            changed = dict(payload)
            changed[key] = value
            with self.assertRaises(ValueError, msg=key):
                readiness.validate_readiness_payload_antigravity(changed)
