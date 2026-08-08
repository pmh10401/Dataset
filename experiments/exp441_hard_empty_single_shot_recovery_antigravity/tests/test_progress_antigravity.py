"""RED/GREEN behavior tests for Exp441 process-owned progress evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity import (
    ProgressWriterAntigravity,
)


_EXACT_FIELDS = {
    "pid", "active", "current_dataset", "current_record_name", "active_datasets",
    "train_row_count", "test_row_count", "raw_length", "dimension_status", "stage",
    "backend", "completed", "total", "imported", "recomputed", "success", "error",
    "provisional_candidates", "outside_hard_candidates", "eligible_candidates",
    "selected_candidates", "throughput", "elapsed_seconds", "eta_seconds",
    "source_checkpoint", "heartbeat", "latest_error",
}


class ProgressAntigravityTests(unittest.TestCase):
    def test_parent_acceptance_writes_atomic_progress_and_append_only_logs(self) -> None:
        # Break caught: a worker-completion event could advance visible state, omit an exact field, or truncate evidence.
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=2, pid=4321)
            writer.begin_record_antigravity(
                dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A", "B"),
            )
            writer.accept_record_antigravity(
                dataset="A", record_name="0000_record_antigravity.json", active_datasets=("B",),
                next_dataset="B", next_record_name="0001_record_antigravity.json",
                test_row_count=17, raw_length=64,
            )
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual(set(progress), _EXACT_FIELDS)
            self.assertTrue(progress["active"])
            self.assertEqual(progress["current_dataset"], "B")
            self.assertEqual(progress["active_datasets"], ["B"])
            self.assertIsNone(progress["train_row_count"])
            self.assertIsNone(progress["raw_length"])
            self.assertEqual(progress["dimension_status"], "not_available_in_frozen_selector_source")
            events = (attempt / "events_antigravity.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["dataset"], "A")
            self.assertIn("accepted dataset=A", (attempt / "text_antigravity.log").read_text())

    def test_terminal_state_clears_active_context_and_marks_pid_inactive(self) -> None:
        # Break caught: terminal success/failure leaves a stale current dataset or live PID that invites forbidden resume.
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=1, pid=4321)
            writer.begin_record_antigravity(
                dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A",),
            )
            for stage, error in (("complete", None), ("failed", "source invalid"), ("interrupted", "SIGTERM received")):
                writer.terminal_antigravity(stage=stage, latest_error=error)
                progress = json.loads((attempt / "progress_antigravity.json").read_text())
                pid = json.loads((attempt / "pid_antigravity").read_text())
                self.assertFalse(progress["active"], stage)
                self.assertIsNone(progress["current_dataset"], stage)
                self.assertIsNone(progress["current_record_name"], stage)
                self.assertEqual(progress["active_datasets"], [], stage)
                self.assertFalse(pid["active"], stage)

    def test_completed_terminal_eta_is_null_for_publication_contract(self) -> None:
        # Break caught: the writer emits numeric zero while the strict publication boundary requires null.
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=1, pid=4321)
            writer.begin_record_antigravity(
                dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A",),
            )
            writer.accept_record_antigravity(
                dataset="A", record_name="0000_record_antigravity.json", active_datasets=(),
                test_row_count=17, raw_length=64,
            )
            writer.source_replay_complete_antigravity()
            writer.selection_counts_antigravity(provisional=1, outside_hard=1, eligible=1, selected=1)
            writer.terminal_antigravity(stage="complete", latest_error=None)
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertIsNone(progress["eta_seconds"])

    def test_short_write_completes_full_json_and_failed_replacement_preserves_previous_json(self) -> None:
        # Break caught: a POSIX short write truncates an artifact, or a failed update replaces the last valid progress JSON.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as progress_module
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=1, pid=4321)
            previous = (attempt / "progress_antigravity.json").read_bytes()
            real_write = progress_module.os.write
            def short_write(descriptor: int, payload: bytes) -> int:
                return real_write(descriptor, payload[:max(1, len(payload) // 7)])
            with patch.object(progress_module.os, "write", side_effect=short_write):
                writer.begin_record_antigravity(dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A",))
            self.assertEqual(json.loads((attempt / "progress_antigravity.json").read_text())["current_dataset"], "A")
            before_failure = (attempt / "progress_antigravity.json").read_bytes()
            with patch.object(progress_module.os, "write", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    writer.begin_record_antigravity(dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A",))
            self.assertNotEqual(before_failure, previous)
            self.assertEqual((attempt / "progress_antigravity.json").read_bytes(), before_failure)
            self.assertFalse((attempt / "progress_antigravity.tmp").exists())

    def test_known_test_and_raw_dimensions_are_recorded_without_inventing_train_data(self) -> None:
        # Break caught: progress hides record-provided TEST/raw dimensions or invents unavailable TRAIN dimensions.
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=1, pid=4321)
            writer.begin_record_antigravity(dataset="A", record_name="0000_record_antigravity.json", active_datasets=("A",))
            writer.accept_record_antigravity(dataset="A", record_name="0000_record_antigravity.json", active_datasets=(), test_row_count=17, raw_length=64)
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((progress["test_row_count"], progress["raw_length"]), (None, None))
            event = json.loads((attempt / "events_antigravity.jsonl").read_text())
            self.assertEqual((event["test_row_count"], event["raw_length"]), (17, 64))
            self.assertIsNone(progress["train_row_count"])
            self.assertEqual(progress["stage"], "summarizing_source_authority")
            self.assertNotIn("verifying", progress["stage"])

    def test_initial_progress_failure_after_pid_creation_marks_pid_inactive_before_reraising(self) -> None:
        # Break caught: a partial writer leaves a live PID if its initial atomic progress artifact cannot be created.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as module
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"; attempt.mkdir()
            with patch.object(module.ProgressWriterAntigravity, "_write_progress_antigravity", side_effect=OSError("initial progress failed")):
                with self.assertRaisesRegex(OSError, "initial progress failed"):
                    ProgressWriterAntigravity(attempt, total=1, pid=4321)
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
            self.assertTrue((attempt / "events_antigravity.jsonl").exists())
            self.assertTrue((attempt / "text_antigravity.log").exists())

    def test_terminal_replace_failure_preserves_progress_cleans_temp_and_still_attempts_pid(self) -> None:
        # Break caught: failed progress replacement prevents PID shutdown or replaces a known-good JSON.
        import experiments.exp441_hard_empty_single_shot_recovery_antigravity.progress_antigravity as module
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"; attempt.mkdir()
            writer = ProgressWriterAntigravity(attempt, total=1, pid=4321)
            prior = (attempt / "progress_antigravity.json").read_bytes()
            real_replace = module.os.replace
            def fail_progress(source, destination):
                if Path(destination).name == "progress_antigravity.json": raise OSError("replace failed")
                return real_replace(source, destination)
            with patch.object(module.os, "replace", side_effect=fail_progress):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    writer.terminal_antigravity(stage="failed", latest_error="source")
            self.assertEqual((attempt / "progress_antigravity.json").read_bytes(), prior)
            self.assertFalse((attempt / "progress_antigravity.tmp").exists())
            self.assertFalse(json.loads((attempt / "pid_antigravity").read_text())["active"])
