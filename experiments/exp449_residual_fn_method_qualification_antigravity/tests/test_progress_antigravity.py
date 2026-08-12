"""Regression coverage for Task 5's create-only progress authority."""
from __future__ import annotations

import json
import os
import inspect
import importlib.util
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


def _session_inventory_antigravity() -> tuple[str, ...]:
    return tuple(f"{sequence:03d}_leaf_antigravity.json" for sequence in range(96))


class ProgressAntigravityTests(unittest.TestCase):
    def test_commit_session_removes_caller_forgeable_receipt_and_raw_progress_authority_antigravity(self) -> None:
        """Break caught: callers can forge progress authority from leaf metadata or a count."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        self.assertFalse(hasattr(progress, "CandidateCommitReceiptAntigravity"))
        self.assertNotIn("return_commit_receipt", inspect.signature(progress.commit_candidate_leaf_antigravity).parameters)
        self.assertFalse(hasattr(progress, "record_verified_progress_antigravity"))
        self.assertFalse(hasattr(progress, "verified_prefix_antigravity"))

    def test_commit_session_rejects_noncontiguous_commit_sequences_antigravity(self) -> None:
        """Break caught: a session records a duplicate, skipped, or nonzero first candidate commit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_commit_session_antigravity_") as temporary:
            root = Path(temporary)
            progress.initialise_attempt_antigravity(root, {"schema": "exp449"})
            session = progress._new_empty_commit_session_antigravity(root, _session_inventory_antigravity())
            with self.assertRaises(ValueError):
                session.commit(1, {"sequence": 1})
            session.commit(0, {"sequence": 0})
            with self.assertRaises(ValueError):
                session.commit(0, {"sequence": 0})
            with self.assertRaises(ValueError):
                session.commit(2, {"sequence": 2})

    def test_commit_session_derives_progress_and_cannot_adopt_external_leaf_antigravity(self) -> None:
        """Break caught: a caller advances progress by count or adopts an independently written canonical leaf."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_commit_session_antigravity_") as temporary:
            root = Path(temporary)
            progress.initialise_attempt_antigravity(root, {"schema": "exp449"})
            inventory = _session_inventory_antigravity()
            session = progress._new_empty_commit_session_antigravity(root, inventory)
            self.assertEqual(tuple(inspect.signature(session.record_progress).parameters), ())
            with self.assertRaises(TypeError):
                session.record_progress(1)
            progress.commit_candidate_leaf_antigravity(root / "candidates_antigravity", inventory[0], {"sequence": 0})
            with self.assertRaises(FileExistsError):
                session.commit(0, {"sequence": 0})
            self.assertEqual(json.loads((root / "progress_antigravity.json").read_text())["dataset_unit_completed"], 0)

    def test_commit_session_requires_exact_96_ordered_inventory_antigravity(self) -> None:
        """Break caught: a session accepts a partial, oversized, duplicate, or unordered commit inventory."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_commit_session_antigravity_") as temporary:
            root = Path(temporary)
            for count in (0, 1, 2, 95, 97):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    progress._new_empty_commit_session_antigravity(root, tuple(f"{sequence:03d}_leaf_antigravity.json" for sequence in range(count)))
            unordered = list(_session_inventory_antigravity()); unordered[0], unordered[1] = unordered[1], unordered[0]
            with self.assertRaises(ValueError):
                progress._new_empty_commit_session_antigravity(root, tuple(unordered))

    def test_map_validation_rejects_nonexistent_or_wrong_map_parent_antigravity(self) -> None:
        """Break caught: a code map claims a non-commit or a map-child's wrong source parent."""
        root = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location("build_codemap_exp449_antigravity", root / "scratch/build_codemap_exp449_antigravity.py")
        self.assertIsNotNone(spec); module = importlib.util.module_from_spec(spec); self.assertIsNotNone(spec.loader); spec.loader.exec_module(module)
        graph = json.loads((root / "docs/codemap/codemap_exp449_antigravity.json").read_text())
        lock = json.loads((root / "docs/codemap/codemap_exp449_antigravity.lock").read_text())
        html = (root / "docs/codemap/codemap_exp449_antigravity.html").read_text()
        def html_for(graph_payload):
            payload = json.dumps(graph_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return re.sub(r'(<script id="codemap-data" type="application/json">).*?(</script>)', r"\1" + payload + r"\2", html)
        graph["generated_from_commit"] = "0" * 40; lock["current_commit"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "source commit"):
            module.validate_map_antigravity(graph, lock, html_for(graph))
        map_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        graph["generated_from_commit"] = map_head; lock["current_commit"] = map_head
        with self.assertRaisesRegex(ValueError, "map parent"):
            module.validate_map_antigravity(graph, lock, html_for(graph))
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
            self.assertEqual(set(payload), {"schema", "primary_error", "cleanup_error", "primary_error_notes", "prefix_validation_error", "exact_next_sequence"})
            self.assertEqual(payload["primary_error"], "ValueError: progress write failed")
            self.assertEqual(payload["cleanup_error"], "OSError: pid close failed")
            self.assertEqual(payload["primary_error_notes"], [])
            self.assertIsNone(payload["prefix_validation_error"])
            with self.assertRaises(FileExistsError):
                write_failure_audit_antigravity(audit, RuntimeError("replacement"), None)

    def test_attempt_channels_advance_only_after_session_commit_antigravity(self) -> None:
        """Break caught: reporting progress before the candidate leaf is durably committed."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            _new_empty_commit_session_antigravity,
            initialise_attempt_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            initialise_attempt_antigravity(root, {"schema": "exp449"})
            session = _new_empty_commit_session_antigravity(root, _session_inventory_antigravity())
            with self.assertRaises(ValueError):
                session.record_progress()
            session.commit(0, {"sequence": 0})
            session.record_progress()
            self.assertEqual(json.loads((root / "progress_antigravity.json").read_text())["dataset_unit_completed"], 1)
            self.assertFalse(json.loads((root / "progress_antigravity.json").read_text())["terminal"])
            self.assertEqual((root / "progress_antigravity.jsonl").read_text().count("\n"), 1)
            self.assertEqual((root / "progress_antigravity.txt").read_text(), "1\n")
            for sequence in range(1, 96):
                session.commit(sequence, {"sequence": sequence})
                session.record_progress()
            self.assertTrue(json.loads((root / "progress_antigravity.json").read_text())["terminal"])

    def test_create_only_leaf_commit_requires_canonical_re_read_antigravity(self) -> None:
        """Break caught: accepting a leaf before its durable canonical reread."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            commit_candidate_leaf_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp449_progress_antigravity_") as temporary:
            root = Path(temporary)
            payload = {"sequence": 0, "candidate_name": "Crop_A", "indices": [1], "contains_test_labels": False}
            digest = commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)
            self.assertEqual(len(digest), 64)
            self.assertEqual(json.loads((root / "000_Crop_A.json").read_text()), payload)
            with self.assertRaises(FileExistsError):
                commit_candidate_leaf_antigravity(root, "000_Crop_A.json", payload)

    def test_session_validate_prefix_returns_longest_prefix_for_all_received_leaf_tampering_antigravity(self) -> None:
        """Break caught: a received leaf defect raises or advances past its first failed sequence."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            _new_empty_commit_session_antigravity,
            initialise_attempt_antigravity,
        )

        for received in (0, 47, 95):
            for defect in ("missing", "replacement", "symlink", "noncanonical", "hash_mismatch"):
                with self.subTest(received=received, defect=defect), tempfile.TemporaryDirectory(prefix="exp449_prefix_antigravity_") as temporary:
                    root = Path(temporary)
                    initialise_attempt_antigravity(root, {"schema": "exp449"})
                    inventory = _session_inventory_antigravity()
                    session = _new_empty_commit_session_antigravity(root, inventory)
                    for sequence in range(received + 1):
                        session.commit(sequence, {"sequence": sequence})
                    leaf = root / "candidates_antigravity" / inventory[received]
                    if defect == "missing":
                        leaf.unlink()
                    elif defect == "replacement":
                        replacement = root / "replacement_antigravity.json"
                        replacement.write_text(f'{{"sequence":{received}}}\n')
                        os.replace(replacement, leaf)
                    elif defect == "symlink":
                        replacement = root / "symlink_target_antigravity.json"
                        replacement.write_text(f'{{"sequence":{received}}}\n')
                        leaf.unlink(); leaf.symlink_to(replacement.name)
                    elif defect == "noncanonical":
                        leaf.write_text(f'{{"sequence": {received}}}\n')
                    else:
                        leaf.write_text(f'{{"sequence":{received},"tampered":true}}\n')
                    result = session.validate_prefix()
                    self.assertEqual(result.valid_prefix, received)
                    self.assertIsInstance(result.error, str)
                    self.assertTrue(result.error)

    def test_prefix_validation_value_is_immutable_slotted_and_ignores_unreturned_official_leaf_antigravity(self) -> None:
        """Break caught: externally shaped namespace state becomes session provenance or mutable validation evidence."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import progress_antigravity as progress  # noqa: PLC0415

        self.assertTrue(hasattr(progress, "PrefixValidationAntigravity"))
        self.assertTrue(hasattr(progress.PrefixValidationAntigravity, "__slots__"))
        with self.assertRaises((AttributeError, TypeError)):
            progress.PrefixValidationAntigravity(0, None).valid_prefix = 1

    def test_session_prefix_ignores_official_shaped_unreturned_next_leaf_at_0_47_95_antigravity(self) -> None:
        """Break caught: canonical official-shaped next-slot JSON is mistaken for a session commit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.progress_antigravity import (  # noqa: PLC0415
            _new_empty_commit_session_antigravity,
            initialise_attempt_antigravity,
        )
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import _candidate_payload_antigravity  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity  # noqa: PLC0415

        inventory = tuple(f"{sequence:03d}_session_antigravity.json" for sequence in range(96))
        for completed in (0, 47, 95):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory(prefix="exp449_session_prefix_antigravity_") as temporary:
                root = Path(temporary)
                initialise_attempt_antigravity(root, {"schema": "exp449"})
                session = _new_empty_commit_session_antigravity(root, inventory)
                for sequence in range(completed):
                    session.commit(sequence, {"sequence": sequence})
                replacement = _candidate_payload_antigravity(completed, "Crop_normal_1", "Crop", "minirocket_local_gap", [], 3, 5, [{"seed": 20260721, "feature_dimension": 1}, {"seed": 20261721, "feature_dimension": 1}])
                (root / "candidates_antigravity" / inventory[completed]).write_bytes(canonical_json_bytes_antigravity(replacement))
                result = session.validate_prefix()
                self.assertEqual(result.valid_prefix, completed)
                self.assertIsNone(result.error)
