"""Regression coverage for Task 5's create-only progress authority."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class ProgressAntigravityTests(unittest.TestCase):
    def test_verified_leaf_writer_removes_exact_created_inode_on_reread_fault_antigravity(self) -> None:
        """Break caught: leaving an unverified or orphaned candidate after a post-create failure."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import commit_candidate_leaf_antigravity  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_leaf_antigravity_") as temporary:
            directory = Path(temporary)
            def fault(tag, _sequence):
                if tag == "cache_reread": raise OSError("reread injected")
            with self.assertRaisesRegex(OSError, "reread injected"):
                commit_candidate_leaf_antigravity(directory, "000_leaf_antigravity.json", {"sequence": 0}, fault_hook=fault)
            self.assertFalse((directory / "000_leaf_antigravity.json").exists())
    def test_failure_audit_is_independent_create_only_fallback_antigravity(self) -> None:
        """Break caught: losing the primary failure when run-root channels cannot be written."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import write_failure_audit_antigravity  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            audit = Path(temporary) / "failure_antigravity.json"
            write_failure_audit_antigravity(audit, ValueError("progress write failed"), OSError("pid close failed"))
            payload = json.loads(audit.read_text())
            self.assertEqual(payload["primary_error"], "ValueError: progress write failed")
            self.assertEqual(payload["cleanup_error"], "OSError: pid close failed")
            with self.assertRaises(FileExistsError):
                write_failure_audit_antigravity(audit, RuntimeError("replacement"), None)

    def test_attempt_channels_advance_only_after_verified_prefix_antigravity(self) -> None:
        """Break caught: reporting progress before the candidate leaf is durably committed."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            initialise_attempt_antigravity,
            record_verified_progress_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            initialise_attempt_antigravity(root, {"schema": "exp449"})
            with self.assertRaises(ValueError):
                record_verified_progress_antigravity(root, ("000_Crop_A.json",), 1)
            commit_candidate_leaf_antigravity(root / "candidates_antigravity", "000_Crop_A.json", {"sequence": 0})
            record_verified_progress_antigravity(root, ("000_Crop_A.json",), 1)
            self.assertEqual(json.loads((root / "progress_antigravity.json").read_text())["dataset_unit_completed"], 1)
            self.assertEqual((root / "progress_antigravity.jsonl").read_text().count("\n"), 1)
            self.assertEqual((root / "progress_antigravity.txt").read_text(), "1\n")

    def test_create_only_leaf_commit_requires_canonical_re_read_antigravity(self) -> None:
        """Break caught: accepting a leaf before its durable canonical reread."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            payload = {"sequence": 0, "candidate_name": "Crop_A", "indices": [1], "contains_test_labels": False}
            digest = commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads((root / "000_Crop_A.json").read_text()), payload)
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)
            with self.assertRaises(FileExistsError):
                commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)

    def test_verified_prefix_stops_at_predecessor_drift_antigravity(self) -> None:
        """Break caught: trusting progress past a missing or altered predecessor leaf."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            commit_candidate_leaf_antigravity(root, "000_Crop_A.json", {"sequence": 0})
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)
            (root / "000_Crop_A.json").write_text('{"sequence":0}\n')
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json")), 1)
