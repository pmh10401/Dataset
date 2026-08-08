"""RED/GREEN behavior tests for Exp441 immutable attempts and runtime ownership."""
from __future__ import annotations

import json
import os
from pathlib import Path
from hashlib import sha256
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.config_antigravity import governance_payload_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import FrozenSourceSnapshotAntigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.sources_antigravity import hard_coordinate_identity_sha256_antigravity
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity import (
    allocate_attempt_antigravity,
    run_exp441_new_antigravity,
    terminal_attempt_authority_sha256_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import (
    select_hard_empty_single_shot_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.tests.test_selector_antigravity import _snapshot, _fixture_candidates


class RuntimeAntigravityTests(unittest.TestCase):
    @staticmethod
    def _prefix(snapshot):
        return snapshot.context_datasets, {}, snapshot.hard_coordinates, {}, {}

    @staticmethod
    def _replay(snapshot):
        def replay(observer):
            for index, dataset in enumerate(snapshot.context_datasets):
                values = {item.raw_length for item in snapshot.provisional_candidates if item.dataset == dataset}
                observer({"sequence": index, "dataset": dataset, "test_row_count": snapshot.test_row_count_by_dataset[dataset], "raw_length": next(iter(values)) if len(values) == 1 else None})
            return snapshot
        return replay

    @staticmethod
    def _attempt_bytes_antigravity(attempt: Path) -> dict[str, str]:
        """Fingerprint every immutable attempt file by its relative path and bytes."""
        return {
            path.relative_to(attempt).as_posix(): sha256(path.read_bytes()).hexdigest()
            for path in sorted(attempt.rglob("*")) if path.is_file()
        }

    def test_attempt_allocation_uses_smallest_gap_and_rejects_non_directory_claim(self) -> None:
        # Break caught: allocating monotonically or treating a symlink/file as a completed immutable attempt.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts_antigravity"
            attempts.mkdir()
            (attempts / "attempt_0000_antigravity").mkdir()
            (attempts / "attempt_0002_antigravity").mkdir()
            allocated = allocate_attempt_antigravity(root)
            self.assertEqual(allocated.name, "attempt_0001_antigravity")
            link = attempts / "attempt_0003_antigravity"
            link.symlink_to(attempts / "attempt_0000_antigravity", target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "attempt"):
                allocate_attempt_antigravity(root)
            link.unlink()
            bad = attempts / "attempt_0003_antigravity"
            bad.write_text("not a directory")
            with self.assertRaisesRegex(ValueError, "attempt"):
                allocate_attempt_antigravity(root)

    def test_terminal_authority_capture_rejects_first_file_mutation_during_later_read(self) -> None:
        # Break caught: sequential closes let a changed first authority file escape the runtime's returned digest.
        names = (
            "source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json",
            "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log",
        )
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_antigravity"; attempt.mkdir(); os.chmod(attempt, 0o700)
            for index, name in enumerate(names):
                (attempt / name).write_bytes(f"file-{index}\n".encode()); os.chmod(attempt / name, 0o600)
            real_read = os.read; reads = {"count": 0}
            def mutate_first_after_eof(descriptor: int, size: int) -> bytes:
                payload = real_read(descriptor, size); reads["count"] += 1
                if reads["count"] == 3:
                    (attempt / names[0]).write_bytes(b"FILE-0\n")
                return payload
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.artifacts_antigravity.os.read", side_effect=mutate_first_after_eof):
                with self.assertRaisesRegex(ValueError, "changed"):
                    terminal_attempt_authority_sha256_antigravity(attempt)

    def test_terminal_authority_capture_revalidates_every_post_hash_mutation(self) -> None:
        # Break caught: a valid grouped read becomes stale while the digest is consumed before return.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import runtime_antigravity
        names = ("source_snapshot_antigravity.json", "selection_antigravity.json", "progress_antigravity.json", "pid_antigravity", "events_antigravity.jsonl", "text_antigravity.log")
        for mutation in ("write", "replace", "unlink", "chmod", "truncate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                attempt = Path(directory) / "attempt_antigravity"; attempt.mkdir(); os.chmod(attempt, 0o700)
                for index, name in enumerate(names): (attempt / name).write_bytes(f"file-{index}\n".encode()); os.chmod(attempt / name, 0o600)
                target = attempt / names[0]; real_hash = runtime_antigravity.terminal_attempt_authority_sha256_from_bytes_antigravity
                def hash_then_mutate(payloads: object) -> str:
                    digest = real_hash(payloads)
                    if mutation == "write": target.write_bytes(b"FILE-0\n")
                    elif mutation == "replace":
                        replacement = attempt / "replacement_antigravity.tmp"; replacement.write_bytes(b"FILE-0\n"); os.chmod(replacement, 0o600); os.replace(replacement, target)
                    elif mutation == "unlink": target.unlink()
                    elif mutation == "chmod": os.chmod(target, 0o644)
                    else: target.write_bytes(b"")
                    return digest
                with patch.object(runtime_antigravity, "terminal_attempt_authority_sha256_from_bytes_antigravity", side_effect=hash_then_mutate):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        terminal_attempt_authority_sha256_antigravity(attempt)

    def test_completed_attempt_constructs_while_root_and_all_authority_descriptors_are_retained(self) -> None:
        # Break caught: returning the terminal hash before construction closes the root and six authority descriptors.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import runtime_antigravity
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import CompletedAttemptAntigravity
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        observed: dict[str, object] = {}
        real_pin = runtime_antigravity.pin_exact_regular_inventory_antigravity
        real_post_init = CompletedAttemptAntigravity.__post_init__

        def track_pin(*args: object, **kwargs: object):
            pinned = real_pin(*args, **kwargs)
            observed["pinned"] = pinned
            return pinned

        def inspect_constructor(instance: CompletedAttemptAntigravity) -> None:
            pinned = observed["pinned"]
            descriptors = (pinned._root_descriptor_antigravity, *pinned._descriptors_antigravity.values())
            observed["descriptors"] = descriptors
            self.assertEqual(len(descriptors), 7)
            for descriptor in descriptors:
                os.fstat(descriptor)
            real_post_init(instance)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime_antigravity, "FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch.object(runtime_antigravity, "_ensure_clean_reviewed_commit_antigravity"), \
                 patch.object(runtime_antigravity, "_load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch.object(runtime_antigravity, "_verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch.object(runtime_antigravity, "_reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch.object(runtime_antigravity, "select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch.object(runtime_antigravity, "validate_production_selection_antigravity", side_effect=lambda value: value), \
                 patch.object(runtime_antigravity, "pin_exact_regular_inventory_antigravity", side_effect=track_pin), \
                 patch.object(CompletedAttemptAntigravity, "__post_init__", new=inspect_constructor):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)
        for descriptor in observed["descriptors"]:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_construction_gap_mutation_cannot_return_a_stale_completed_attempt(self) -> None:
        # Break caught: an authority mutation synchronized during completed-attempt construction escapes after the old hash helper returns.
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import runtime_antigravity
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import CompletedAttemptAntigravity
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        observed: dict[str, object] = {}
        construction_entered = threading.Event()
        mutation_finished = threading.Event()
        mutation_errors: list[BaseException] = []
        real_pin = runtime_antigravity.pin_exact_regular_inventory_antigravity
        real_post_init = CompletedAttemptAntigravity.__post_init__

        def track_pin(*args: object, **kwargs: object):
            pinned = real_pin(*args, **kwargs)
            observed["attempt"] = pinned._root_antigravity
            return pinned

        def mutate_during_construction() -> None:
            try:
                self.assertTrue(construction_entered.wait(timeout=5))
                (observed["attempt"] / "source_snapshot_antigravity.json").write_bytes(b"mutated\n")
            except BaseException as error:
                mutation_errors.append(error)
            finally:
                mutation_finished.set()

        def synchronize_constructor(instance: CompletedAttemptAntigravity) -> None:
            real_post_init(instance)
            construction_entered.set()
            self.assertTrue(mutation_finished.wait(timeout=5))

        worker = threading.Thread(target=mutate_during_construction)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(runtime_antigravity, "_RUN_ROOT_ANTIGRAVITY", root), \
                     patch.object(runtime_antigravity, "FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                     patch.object(runtime_antigravity, "_ensure_clean_reviewed_commit_antigravity"), \
                     patch.object(runtime_antigravity, "_load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                     patch.object(runtime_antigravity, "_verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                     patch.object(runtime_antigravity, "_reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                     patch.object(runtime_antigravity, "select_hard_empty_single_shot_antigravity", return_value=selection), \
                     patch.object(runtime_antigravity, "validate_production_selection_antigravity", side_effect=lambda value: value), \
                     patch.object(runtime_antigravity, "pin_exact_regular_inventory_antigravity", side_effect=track_pin), \
                     patch.object(CompletedAttemptAntigravity, "__post_init__", new=synchronize_constructor):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        run_exp441_new_antigravity(reviewed_commit="a" * 40)
        finally:
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(mutation_errors, [])

    def test_completed_attempt_is_create_only_and_contains_exact_governance(self) -> None:
        # Break caught: a later new could repair an old attempt, or selection evidence could dilute governance values.
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                first = run_exp441_new_antigravity(reviewed_commit="a" * 40)
                first_bytes = (root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_bytes()
                second = run_exp441_new_antigravity(reviewed_commit="a" * 40)
            self.assertEqual((first.attempt_sequence, second.attempt_sequence), (0, 1))
            self.assertEqual(
                first.terminal_authority_sha256,
                terminal_attempt_authority_sha256_antigravity(root / "attempts_antigravity" / "attempt_0000_antigravity"),
            )
            self.assertEqual(
                second.terminal_authority_sha256,
                terminal_attempt_authority_sha256_antigravity(root / "attempts_antigravity" / "attempt_0001_antigravity"),
            )
            selection_payload = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_text())
            governance = governance_payload_antigravity()
            self.assertEqual({name: selection_payload[name] for name in governance}, governance)
            self.assertEqual(set(selection_payload) & set(governance), set(governance))
            self.assertEqual(len(list((root / "attempts_antigravity").iterdir())), 2)
            self.assertEqual((root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json").read_bytes(), first_bytes)

    def test_source_failure_is_immutable_failed_attempt_without_publication(self) -> None:
        # Break caught: source/schema failure escapes without preserved evidence or accidentally publishes a prediction.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", side_effect=ValueError("bad source")), \
                patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=ValueError("bad source")):
                with self.assertRaisesRegex(ValueError, "bad source"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
                with self.assertRaisesRegex(ValueError, "bad source"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertEqual({name: failure[name] for name in governance_payload_antigravity()}, governance_payload_antigravity())
            self.assertFalse((root / "publication_antigravity").exists())
            self.assertTrue((root / "attempts_antigravity" / "attempt_0001_antigravity" / "failure_antigravity.json").exists())

    def test_ordered_parent_observer_rejects_a_later_context_record(self) -> None:
        # Break caught: committing a worker-completed B before the parent has accepted ordered A.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def out_of_order(observer):
                observer({"sequence": 1, "dataset": "B", "test_row_count": 8, "raw_length": None})
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=out_of_order):
                with self.assertRaisesRegex(ValueError, "context order"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            progress = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json").read_text())
            self.assertEqual(progress["stage"], "failed")
            self.assertEqual(progress["completed"], 0)

    def test_progress_stays_unsubmitted_until_source_parser_emits_first_parent_event(self) -> None:
        # Break caught: the runtime reports all datasets active before the source parser has created any futures.
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def replay(observer):
                attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
                before = json.loads((attempt / "progress_antigravity.json").read_text())
                self.assertEqual((before["stage"], before["current_dataset"], before["current_record_name"], before["active_datasets"]), ("allocated", None, None, []))
                observer({"sequence": 0, "dataset": snapshot.context_datasets[0], "test_row_count": 8, "raw_length": 5})
                after_first = json.loads((attempt / "progress_antigravity.json").read_text())
                self.assertEqual((after_first["completed"], after_first["current_dataset"], after_first["active_datasets"]), (1, snapshot.context_datasets[1], list(snapshot.context_datasets[1:])))
                self.assertEqual(len((attempt / "events_antigravity.jsonl").read_text().splitlines()), 1)
                for index, dataset in enumerate(snapshot.context_datasets[1:], start=1):
                    observer({"sequence": index, "dataset": dataset, "test_row_count": 8, "raw_length": None})
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=replay), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)

    def test_sigterm_is_recorded_as_interrupted_not_schema_failure(self) -> None:
        # Break caught: SIGTERM follows the generic failure path and loses the distinct interruption meaning.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def interrupt(_observer):
                os.kill(os.getpid(), 15)
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "SIGTERM"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").read_text())
            self.assertEqual(failure["status"], "interrupted")

    def test_sigterm_handler_is_restored_after_interrupted_attempt(self) -> None:
        # Break caught: the runtime leaves its private SIGTERM handler installed after returning to its caller.
        snapshot = _snapshot(_fixture_candidates())
        prior = signal.getsignal(signal.SIGTERM)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def interrupt(_observer):
                os.kill(os.getpid(), signal.SIGTERM)
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "SIGTERM"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
        self.assertIs(signal.getsignal(signal.SIGTERM), prior)

    def test_writer_initialization_failure_keeps_original_error_and_writes_failure_when_independent_channel_works(self) -> None:
        # Break caught: a writer construction error prevents the independent immutable failure record from being created.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime, "ProgressWriterAntigravity", side_effect=OSError("writer initialization failed")):
                with self.assertRaisesRegex(OSError, "writer initialization failed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure_path = root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json"
            failure = runtime.parse_canonical_json_bytes_antigravity(failure_path.read_bytes())
            runtime._validate_failure_artifact_antigravity(failure)
            self.assertEqual(failure["error"], "writer initialization failed")

    def test_runtime_failure_keeps_primary_error_when_terminal_progress_write_fails(self) -> None:
        # Break caught: an evidence-write failure hides the actual source failure or skips PID/failure evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as progress
        snapshot = _snapshot(_fixture_candidates())
        original = progress.ProgressWriterAntigravity._write_progress_antigravity
        def fail_only_terminal(writer):
            if writer._state["stage"] == "failed":
                raise OSError("terminal progress failed")
            return original(writer)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=ValueError("source failure")), \
                 patch.object(progress.ProgressWriterAntigravity, "_write_progress_antigravity", new=fail_only_terminal):
                with self.assertRaisesRegex(ValueError, "source failure"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
            self.assertEqual(json.loads((attempt / "failure_antigravity.json").read_text())["error"], "source failure")

    def test_runtime_failure_keeps_primary_error_when_terminal_pid_replace_fails(self) -> None:
        # Break caught: a PID-replace fault masks the source error or deletes failed progress/failure evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as progress
        snapshot = _snapshot(_fixture_candidates())
        real_replace = progress.os.replace
        def fail_pid_replace(source, destination):
            if Path(destination).name == "pid_antigravity":
                raise OSError("terminal PID replace failed")
            return real_replace(source, destination)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=ValueError("source failure")), \
                 patch.object(progress.os, "replace", side_effect=fail_pid_replace):
                with self.assertRaisesRegex(ValueError, "source failure"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertEqual(json.loads((attempt / "progress_antigravity.json").read_text())["stage"], "failed")
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
            self.assertEqual(json.loads((attempt / "failure_antigravity.json").read_text())["error"], "source failure")

    def test_partial_writer_initial_progress_failure_preserves_original_error_inactive_pid_and_failure_json(self) -> None:
        # Break caught: a production new attempt leaves a live PID or loses canonical failure evidence after initial progress creation fails.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as progress
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(progress.ProgressWriterAntigravity, "_write_progress_antigravity", side_effect=OSError("initial progress failed")):
                with self.assertRaisesRegex(OSError, "initial progress failed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
            failure = runtime.parse_canonical_json_bytes_antigravity((attempt / "failure_antigravity.json").read_bytes())
            runtime._validate_failure_artifact_antigravity(failure)
            self.assertFalse((root / "publication_antigravity").exists())

    def test_partial_writer_event_or_text_creation_failure_keeps_primary_error_inactive_pid_and_canonical_failure(self) -> None:
        # Break caught: a log-file creation fault after PID creation leaves a live attempt or prevents independent failure evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as progress
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        real_create = progress.write_create_only_antigravity
        for failed_name in ("events_antigravity.jsonl", "text_antigravity.log"):
            with self.subTest(failed_name=failed_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                def fail_one_log(path, payload):
                    if Path(path).name == failed_name:
                        raise OSError(f"{failed_name} creation failed")
                    return real_create(path, payload)
                with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                     patch.object(progress, "write_create_only_antigravity", side_effect=fail_one_log):
                    with self.assertRaisesRegex(OSError, f"{failed_name} creation failed"):
                        run_exp441_new_antigravity(reviewed_commit="a" * 40)
                attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
                self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
                failure = runtime.parse_canonical_json_bytes_antigravity((attempt / "failure_antigravity.json").read_bytes())
                runtime._validate_failure_artifact_antigravity(failure)
                self.assertFalse((root / "publication_antigravity").exists())
                if failed_name == "text_antigravity.log":
                    self.assertEqual((attempt / "events_antigravity.jsonl").read_bytes(), b"")

    def test_runtime_failure_keeps_primary_error_when_failure_json_write_fails(self) -> None:
        # Break caught: a failure-artifact I/O fault masks the source error or rolls back failed progress/PID evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        snapshot = _snapshot(_fixture_candidates())
        real_create = runtime.write_create_only_antigravity
        def fail_only_failure(path, payload):
            if Path(path).name == "failure_antigravity.json":
                raise OSError("failure JSON write failed")
            return real_create(path, payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime, "_ensure_clean_reviewed_commit_antigravity"), \
                 patch.object(runtime, "_load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch.object(runtime, "_verify_with_observer_antigravity", side_effect=ValueError("source failure")), \
                 patch.object(runtime, "write_create_only_antigravity", side_effect=fail_only_failure):
                with self.assertRaisesRegex(ValueError, "source failure"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertEqual(json.loads((attempt / "progress_antigravity.json").read_text())["stage"], "failed")
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
            self.assertFalse((attempt / "failure_antigravity.json").exists())

    def test_governance_validator_rejects_missing_extra_wrong_and_nested_values(self) -> None:
        # Break caught: a governing terminal artifact accepts a partial alias, bool-like substitute, or nested replacement.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        valid = governance_payload_antigravity()
        runtime._validate_governance_payload_antigravity(valid)
        for name in valid:
            missing = dict(valid)
            missing.pop(name)
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(missing)
            wrong = dict(valid)
            wrong[name] = {"value": valid[name]}
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(wrong)
            wrong_value = dict(valid)
            wrong_value[name] = (not valid[name]) if type(valid[name]) is bool else "changed"
            with self.assertRaisesRegex(ValueError, "governance"):
                runtime._validate_governance_payload_antigravity(wrong_value)
        extra = dict(valid)
        extra["review_status"] = "alias"
        with self.assertRaisesRegex(ValueError, "governance"):
            runtime._validate_governance_payload_antigravity(extra)

    def test_real_terminal_artifact_shapes_reject_each_governance_tamper(self) -> None:
        # Break caught: selection/failure JSON accepts a governance omission, alias, wrong primitive, or nested object.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        artifacts = (
            runtime._selection_payload_antigravity(selection, reviewed_commit="a" * 40, source_snapshot_sha256="b" * 64),
            runtime._failure_payload_antigravity(sequence=0, reviewed_commit="a" * 40, status="failed", error="bad source"),
        )
        governance = governance_payload_antigravity()
        for artifact in artifacts:
            runtime._validate_terminal_artifact_governance_antigravity(artifact)
            for name in governance:
                missing = dict(artifact)
                missing.pop(name)
                with self.assertRaisesRegex(ValueError, "governance"):
                    runtime._validate_terminal_artifact_governance_antigravity(missing)
                wrong = dict(artifact)
                wrong[name] = {"nested": governance[name]}
                with self.assertRaisesRegex(ValueError, "governance"):
                    runtime._validate_terminal_artifact_governance_antigravity(wrong)
            alias = dict(artifact)
            alias["review_only_alias"] = True
            with self.assertRaisesRegex(ValueError, "alias"):
                runtime._validate_terminal_artifact_governance_antigravity(alias)

    def test_persisted_selection_and_failure_reopen_with_exact_full_schemas(self) -> None:
        # Break caught: a canonical file reopens while accepting a missing, extra, wrong, nested, status, commit, or hash field.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime, "FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch.object(runtime, "_ensure_clean_reviewed_commit_antigravity"), \
                 patch.object(runtime, "_load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch.object(runtime, "_verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch.object(runtime, "_reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch.object(runtime, "select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch.object(runtime, "validate_production_selection_antigravity", side_effect=lambda value: value):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)
            selection_path = root / "attempts_antigravity" / "attempt_0000_antigravity" / "selection_antigravity.json"
            selection_payload = runtime.parse_canonical_json_bytes_antigravity(selection_path.read_bytes())
            runtime._validate_selection_artifact_antigravity(selection_payload)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime, "_ensure_clean_reviewed_commit_antigravity"), \
                 patch.object(runtime, "_load_prefix_antigravity", side_effect=ValueError("source failure")):
                with self.assertRaisesRegex(ValueError, "source failure"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            failure_path = root / "attempts_antigravity" / "attempt_0001_antigravity" / "failure_antigravity.json"
            failure_payload = runtime.parse_canonical_json_bytes_antigravity(failure_path.read_bytes())
            runtime._validate_failure_artifact_antigravity(failure_payload)
        for payload, validator, mutations in (
            (selection_payload, runtime._validate_selection_artifact_antigravity, (
                lambda value: value.pop("winners"),
                lambda value: value.update({"arbitrary": True}),
                lambda value: value.update({"reviewed_commit": "z" * 40}),
                lambda value: value.update({"source_snapshot_sha256": "z" * 64}),
                lambda value: value.update({"selected": {"nested": True}}),
            )),
            (failure_payload, runtime._validate_failure_artifact_antigravity, (
                lambda value: value.pop("error"),
                lambda value: value.update({"arbitrary": True}),
                lambda value: value.update({"status": "complete"}),
                lambda value: value.update({"reviewed_commit": "z" * 40}),
                lambda value: value.update({"error": {"nested": True}}),
            )),
        ):
            for mutate in mutations:
                tampered = dict(payload)
                mutate(tampered)
                with self.assertRaises(ValueError):
                    validator(tampered)

    def test_failed_and_interrupted_attempts_remain_byte_identical_when_new_allocates_next_sequence(self) -> None:
        # Break caught: a later new repairs an old failed/interrupted directory instead of allocating a fresh immutable attempt.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", side_effect=ValueError("failed source")):
                with self.assertRaisesRegex(ValueError, "failed source"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            first = root / "attempts_antigravity" / "attempt_0000_antigravity"
            first_before = self._attempt_bytes_antigravity(first)
            def interrupt(_observer):
                os.kill(os.getpid(), signal.SIGTERM)
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "SIGTERM"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            second = root / "attempts_antigravity" / "attempt_0001_antigravity"
            second_before = self._attempt_bytes_antigravity(second)
            self.assertEqual(self._attempt_bytes_antigravity(first), first_before)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", side_effect=ValueError("third failure")):
                with self.assertRaisesRegex(ValueError, "third failure"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            self.assertTrue((root / "attempts_antigravity" / "attempt_0002_antigravity").is_dir())
            self.assertEqual(self._attempt_bytes_antigravity(first), first_before)
            self.assertEqual(self._attempt_bytes_antigravity(second), second_before)
            self.assertFalse((first / "selection_antigravity.json").exists())
            self.assertFalse((second / "selection_antigravity.json").exists())

    def test_successful_runtime_calls_replay_selection_artifacts_replay_clean_then_complete(self) -> None:
        # Break caught: the runtime writes either artifact before selection or completes before its final authority/HEAD checks.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        snapshot = _snapshot(_fixture_candidates())
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        calls: list[str] = []
        real_write = runtime.write_create_only_antigravity
        real_terminal = runtime.ProgressWriterAntigravity.terminal_antigravity
        def clean(_commit): calls.append("clean")
        def first_replay(observer):
            calls.append("replay")
            return self._replay(snapshot)(observer)
        def choose(value):
            calls.append("selection")
            return selection
        def write(path, payload):
            calls.append(Path(path).name)
            return real_write(path, payload)
        def reverify():
            calls.append("second_replay")
            return snapshot
        def terminal(writer, *, stage, latest_error):
            calls.append(stage)
            return real_terminal(writer, stage=stage, latest_error=latest_error)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runtime, "_RUN_ROOT_ANTIGRAVITY", root), \
                 patch.object(runtime, "FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch.object(runtime, "_ensure_clean_reviewed_commit_antigravity", side_effect=clean), \
                 patch.object(runtime, "_load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch.object(runtime, "_verify_with_observer_antigravity", side_effect=first_replay), \
                 patch.object(runtime, "select_hard_empty_single_shot_antigravity", side_effect=choose), \
                 patch.object(runtime, "validate_production_selection_antigravity", side_effect=lambda value: value), \
                 patch.object(runtime, "write_create_only_antigravity", side_effect=write), \
                 patch.object(runtime, "_reverify_sources_after_artifacts_antigravity", side_effect=reverify), \
                 patch.object(runtime.ProgressWriterAntigravity, "terminal_antigravity", new=terminal):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)
        self.assertEqual(calls, ["clean", "replay", "selection", "source_snapshot_antigravity.json", "selection_antigravity.json", "second_replay", "clean", "complete"])

    def test_production_shaped_parent_observer_reaches_1117_terminal_counts_and_rechecks_sources(self) -> None:
        # Break caught: a mocked successful source verifier skips parent events, misses the full prefix, or lacks the second authority check.
        datasets = tuple(f"D{index:04d}" for index in range(1117))
        snapshot = FrozenSourceSnapshotAntigravity(
            context_datasets=datasets, test_row_count_by_dataset={name: 8 for name in datasets},
            record_digest_by_dataset={name: f"{index:064x}" for index, name in enumerate(datasets)},
            hard_coordinates=frozenset(), provisional_candidates=(), context_sha256="a" * 64,
            state_sha256="b" * 64, retained_inventory_sha256="c" * 64,
            publication_prediction_sha256="d" * 64, publication_manifest_sha256="e" * 64,
            publication_artifact_map_sha256="f" * 64, hard_prediction_sha256="0" * 64,
            hard_manifest_sha256="1" * 64,
            hard_coordinate_identity_sha256=hard_coordinate_identity_sha256_antigravity(()),
        )
        sparse_selection = select_hard_empty_single_shot_antigravity(_snapshot(_fixture_candidates()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def replay(observer):
                for index, dataset in enumerate(datasets):
                    observer({"sequence": index, "dataset": dataset, "test_row_count": 8, "raw_length": None})
                return snapshot
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity") as clean, \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=replay), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=snapshot), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=sparse_selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                run_exp441_new_antigravity(reviewed_commit="a" * 40)
            progress = json.loads((root / "attempts_antigravity" / "attempt_0000_antigravity" / "progress_antigravity.json").read_text())
            self.assertEqual((progress["completed"], progress["imported"], progress["success"]), (1117, 1117, 1117))
            self.assertEqual((progress["current_dataset"], progress["current_record_name"], progress["active_datasets"]), (None, None, []))
            self.assertEqual((progress["stage"], progress["provisional_candidates"], progress["outside_hard_candidates"], progress["eligible_candidates"], progress["selected_candidates"]), ("complete", 5, 5, 5, 4))
            self.assertEqual(clean.call_count, 2)

    def test_reverification_detects_source_identity_change_after_attempt_artifacts(self) -> None:
        # Break caught: an input mutation after selection is accepted because only the first frozen-source replay is checked.
        snapshot = _snapshot(_fixture_candidates())
        changed = _snapshot(_fixture_candidates())
        object.__setattr__(changed, "context_sha256", "9" * 64)
        selection = select_hard_empty_single_shot_antigravity(snapshot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._reverify_sources_after_artifacts_antigravity", return_value=changed), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", return_value=selection), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.validate_production_selection_antigravity", side_effect=lambda value: value):
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            self.assertTrue((root / "attempts_antigravity" / "attempt_0000_antigravity" / "source_snapshot_antigravity.json").exists())
            self.assertTrue((root / "attempts_antigravity" / "attempt_0000_antigravity" / "failure_antigravity.json").exists())

    def test_selection_failure_writes_no_snapshot_or_selection_artifact(self) -> None:
        # Break caught: source evidence is frozen before the pure selector has successfully completed.
        snapshot = _snapshot(_fixture_candidates())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._RUN_ROOT_ANTIGRAVITY", root), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.FROZEN_CONFIG", SimpleNamespace(dataset_count=len(snapshot.context_datasets))), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._ensure_clean_reviewed_commit_antigravity"), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._load_prefix_antigravity", return_value=self._prefix(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity._verify_with_observer_antigravity", side_effect=self._replay(snapshot)), \
                 patch("experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity.select_hard_empty_single_shot_antigravity", side_effect=ValueError("selection failed")):
                with self.assertRaisesRegex(ValueError, "selection failed"):
                    run_exp441_new_antigravity(reviewed_commit="a" * 40)
            attempt = root / "attempts_antigravity" / "attempt_0000_antigravity"
            self.assertFalse((attempt / "source_snapshot_antigravity.json").exists())
            self.assertFalse((attempt / "selection_antigravity.json").exists())

    def test_source_snapshot_records_only_authority_identity_counts_and_ranking_coordinates(self) -> None:
        # Break caught: an attempt snapshot stores raw source rows, TEST sizes, or any field beyond allowed replay evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        payload = runtime._snapshot_payload_antigravity(_snapshot(_fixture_candidates()))
        self.assertEqual(set(payload), {
            "context_dataset_count", "record_digest_count", "record_digest_map_identity_sha256",
            "hard_coordinate_count", "hard_coordinate_identity_sha256", "provisional_candidate_count",
            "ranking_evidence", "source_hashes", "label_access_count",
        })
        self.assertTrue(all(set(row) == {"dataset", "instance_index", "second_excess_ratio", "second_empirical_rank"}
                            for row in payload["ranking_evidence"]))
        self.assertNotIn("test_row_count_by_dataset", payload)
        self.assertNotIn("provisional_candidates", payload)

    def test_no_resume_api_is_exported(self) -> None:
        # Break caught: exposing a resume entrypoint allows mutation of immutable failure evidence.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.runtime_antigravity as runtime
        self.assertFalse(any("resume" in name.lower() for name in dir(runtime)))
