"""Regression coverage for Task 5's create-only progress authority."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


class ProgressAntigravityTests(unittest.TestCase):
    def test_writer_actual_syscall_faults_leave_no_unverified_orphan_antigravity(self) -> None:
        """Break caught: a create/write/fsync/read/stat syscall failure leaves a candidate namespace artifact."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import commit_candidate_leaf_antigravity  # noqa: PLC0415
        for tag in ("open_create", "fstat_create", "write", "fsync_leaf", "open_read", "read", "fstat_read", "stat_name"):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory(prefix="exp449_syscall_antigravity_") as temporary:
                directory = Path(temporary)
                def fault(current, _sequence):
                    if current == tag: raise OSError(tag)
                with self.assertRaisesRegex(OSError, tag):
                    commit_candidate_leaf_antigravity(directory, "000_leaf_antigravity.json", {"sequence": 0}, fault_hook=fault)
                self.assertFalse((directory / "000_leaf_antigravity.json").exists())

    def test_writer_real_fstat_syscall_failure_preserves_an_unprovable_name_and_reports_cleanup_antigravity(self) -> None:
        """Break caught: treating a pathname as proof of the O_EXCL inode after fstat fails."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_fstat_syscall_antigravity_") as temporary:
            directory = Path(temporary); destination = directory / "000_leaf_antigravity.json"
            with patch.object(progress.os, "fstat", side_effect=OSError("actual fstat failure")):
                with self.assertRaisesRegex(OSError, "actual fstat failure") as raised:
                    progress.commit_candidate_leaf_antigravity(directory, "000_leaf_antigravity.json", {"sequence": 0})
            self.assertTrue(destination.exists())
            self.assertIn("candidate_cleanup_error=OSError: unable to prove created inode after fstat failure", "\n".join(getattr(raised.exception, "__notes__", ())))

    def test_writer_replacement_between_exclusive_open_and_fstat_is_never_deleted_antigravity(self) -> None:
        """Break caught: an open-to-fstat replacement is mistaken for the created inode during cleanup."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_pre_fstat_replace_antigravity_") as temporary:
            directory = Path(temporary); destination = directory / "000_leaf_antigravity.json"; original_fstat = progress.os.fstat; replaced = False
            def replace_before_fstat(fd):
                nonlocal replaced
                if not replaced:
                    replaced = True
                    replacement = directory / "replacement_antigravity.json"; replacement.write_text('{"replacement":true}\n')
                    os.replace(replacement, destination)
                    raise OSError("fstat failure after replacement")
                return original_fstat(fd)
            with patch.object(progress.os, "fstat", side_effect=replace_before_fstat):
                with self.assertRaisesRegex(OSError, "fstat failure after replacement"):
                    progress.commit_candidate_leaf_antigravity(directory, destination.name, {"sequence": 0})
            self.assertTrue(replaced)
            self.assertEqual(json.loads(destination.read_text()), {"replacement": True})

    def test_writer_does_not_unlink_namespace_replacement_after_verification_fault_antigravity(self) -> None:
        """Break caught: cleanup deleting a replacement object that was not the created inode."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import commit_candidate_leaf_antigravity  # noqa: PLC0415
        with tempfile.TemporaryDirectory(prefix="exp449_replace_antigravity_") as temporary:
            directory = Path(temporary); destination = directory / "000_leaf_antigravity.json"
            def fault(tag, _sequence):
                if tag == "cache_reread":
                    replacement = directory / "replacement_antigravity.json"; replacement.write_text('{"replacement":true}\n')
                    os.replace(replacement, destination); raise OSError("reread replacement")
            with self.assertRaisesRegex(OSError, "reread replacement"):
                commit_candidate_leaf_antigravity(directory, destination.name, {"sequence": 0}, fault_hook=fault)
            self.assertEqual(json.loads(destination.read_text()), {"replacement": True})
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
            self.assertEqual(set(payload), {"schema", "primary_error", "cleanup_error", "primary_error_notes", "exact_next_sequence"})
            self.assertEqual(payload["primary_error"], "ValueError: progress write failed")
            self.assertEqual(payload["cleanup_error"], "OSError: pid close failed")
            self.assertEqual(payload["primary_error_notes"], [])
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
                record_verified_progress_antigravity(root, ("000_Crop_A.json",), (), 1)
            receipt = commit_candidate_leaf_antigravity(root / "candidates_antigravity", "000_Crop_A.json", {"sequence": 0}, return_commit_receipt=True)
            record_verified_progress_antigravity(root, ("000_Crop_A.json",), (receipt,), 1)
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
            receipt = commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload, return_commit_receipt=True)
            self.assertEqual(len(receipt.sha256), 64)
            self.assertEqual(json.loads((root / "000_Crop_A.json").read_text()), payload)
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json"), (receipt,)), 1)
            with self.assertRaises(FileExistsError):
                commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)

    def test_receipt_bound_prefix_rejects_received_leaf_replacement_antigravity(self) -> None:
        """Break caught: trusting a canonical pathname after its returned receipt was replaced."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            receipt = commit_candidate_leaf_antigravity(root, "000_Crop_A.json", {"sequence": 0}, return_commit_receipt=True)
            self.assertEqual(verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json"), (receipt,)), 1)
            replacement = root / "replacement_antigravity.json"
            replacement.write_text('{"sequence":0}\n')
            os.replace(replacement, root / "000_Crop_A.json")
            with self.assertRaisesRegex(ValueError, "receipt"):
                verified_prefix_antigravity(root, ("000_Crop_A.json", "001_Crop_B.json"), (receipt,))

    def test_receipt_bound_prefix_ignores_official_shaped_unreturned_next_leaf_at_0_47_95_antigravity(self) -> None:
        """Break caught: canonical official-shaped next-slot JSON is mistaken for a returned commit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
            verified_prefix_antigravity,
        )
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import _candidate_payload_antigravity  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity  # noqa: PLC0415

        inventory = tuple(f"{sequence:03d}_receipt_antigravity.json" for sequence in range(96))
        for completed in (0, 47, 95):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory(prefix="exp449_receipt_prefix_antigravity_") as temporary:
                root = Path(temporary); receipts = []
                for sequence in range(completed):
                    receipt = commit_candidate_leaf_antigravity(root, inventory[sequence], {"sequence": sequence}, sequence=sequence, return_commit_receipt=True)
                    receipts.append(receipt)
                replacement = _candidate_payload_antigravity(completed, "Crop_normal_1", "Crop", "minirocket_local_gap", [], 3, 5, [{"seed": 20260721, "feature_dimension": 1}, {"seed": 20261721, "feature_dimension": 1}])
                (root / inventory[completed]).write_bytes(canonical_json_bytes_antigravity(replacement))
                self.assertEqual(verified_prefix_antigravity(root, inventory, tuple(receipts)), completed)
