"""TDD tests for the Exp444 self-written progress/PID evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


class Exp444ProgressAntigravityTests(unittest.TestCase):
    def test_progress_records_current_dimensions_and_eta_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            progress = attempt / "progress_antigravity.json"
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42001, checkpoint_path=attempt / "checkpoint_antigravity/state_antigravity.json")
            writer.start_candidate_antigravity("A", 7, train_count=20, test_count=50, raw_length=144)
            writer.start_view_antigravity(view_index=3, view_count=16, excluded_count=2)
            state = json.loads(progress.read_text())
            self.assertEqual(state["current_dataset"], "A")
            self.assertEqual(state["current_candidate_index"], 7)
            self.assertEqual(state["current_view"], 3)
            self.assertEqual(state["raw_length"], 144)
            self.assertEqual(state["view_count"], 16)
            self.assertIsNotNone(state["eta_seconds"])
            self.assertEqual(state["error"], 0)

    def test_completion_clears_dimensions_and_terminal_eta_and_logs_atomically_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=1, pid=42002)
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("complete")
            progress_bytes = (attempt / "progress_antigravity.json").read_bytes()
            state = json.loads(progress_bytes)
            self.assertFalse(state["active"])
            self.assertEqual(state["stage"], "complete")
            self.assertIsNone(state["current_dataset"])
            self.assertIsNone(state["current_candidate_index"])
            self.assertIsNone(state["current_view"])
            self.assertIsNone(state["eta_seconds"])
            self.assertEqual(state["completed"], 1)
            pid = json.loads((attempt / "pid_antigravity.json").read_text())
            self.assertFalse(pid["active"])
            events = (attempt / "events_antigravity.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(events[-1])["event"], "terminal")
            self.assertIn("종료 stage=complete", (attempt / "text_antigravity.log").read_text())
            before = progress_bytes
            with self.assertRaisesRegex(ValueError, "terminal"):
                writer.start_candidate_antigravity("B", 0, train_count=1, test_count=1, raw_length=1)
            self.assertEqual(before, (attempt / "progress_antigravity.json").read_bytes())

    def test_interrupted_progress_restores_counters_and_rejects_live_resume_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=2, pid=42003, start_identity="start-A")
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            with self.assertRaisesRegex(ValueError, "active"):
                ProgressWriterAntigravity.open_interrupted(attempt, total=2, pid=42004, start_identity="start-B", pid_probe=lambda _: True)
            resumed = ProgressWriterAntigravity.open_interrupted(attempt, total=2, pid=42004, start_identity="start-B", pid_probe=lambda _: False)
            self.assertEqual(resumed.completed_count_antigravity, 1)
            self.assertEqual(resumed.error_count_antigravity, 0)
            self.assertEqual(json.loads((attempt / "progress_antigravity.json").read_text())["active"], True)


if __name__ == "__main__":
    unittest.main()
