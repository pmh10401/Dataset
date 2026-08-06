"""TDD checks for Exp435 progress, events, and PID lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.progress_antigravity import (
    ProgressWriter,
)


class ProgressWriterTests(unittest.TestCase):
    def test_atomic_progress_contains_required_operational_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            writer = ProgressWriter.create(
                root,
                run_identity="exp435-fixture",
                checkpoint=Path(directory) / "checkpoint",
                total_datasets=2,
                backend="cpu",
                heartbeat_seconds=None,
            )
            writer.update(
                stage="scoring",
                current_dataset="Demo",
                dataset_metadata={
                    "train_size": 4,
                    "train_lengths": [8],
                    "test_size": 2,
                    "test_lengths": [8],
                },
                completed=1,
                success=1,
                abstention=0,
                error=0,
                candidate_count=3,
            )
            payload = json.loads((root / "progress_antigravity.json").read_text())
            for field in (
                "pid", "heartbeat", "current_dataset", "dataset_metadata", "stage",
                "backend", "completed", "total", "success", "abstention", "error",
                "candidate_count", "throughput", "elapsed_seconds", "eta_seconds",
                "checkpoint", "latest_error",
            ):
                self.assertIn(field, payload)
            self.assertEqual(payload["current_dataset"], "Demo")
            writer.close()
            pid = json.loads((root / "run_antigravity.pid").read_text())
            self.assertFalse(pid["active"])

    def test_resume_rejects_live_pid_and_allows_inactive_cleanup(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            writer = ProgressWriter.create(
                root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                total_datasets=1, backend="cpu", heartbeat_seconds=None,
            )
            with self.assertRaisesRegex(ValueError, "active"):
                ProgressWriter.open_existing(
                    root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                    process_is_live=lambda pid: True,
                )
            writer.close()
            resumed = ProgressWriter.open_existing(
                root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                process_is_live=lambda pid: False,
            )
            resumed.close()

    def test_failure_and_interruption_are_durable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            writer = ProgressWriter.create(
                root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                total_datasets=1, backend="cpu", heartbeat_seconds=None,
            )
            writer.mark_interrupted("Demo", RuntimeError("stopped"))
            writer.close()
            events = [json.loads(line) for line in (root / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertTrue(any(item["state"] == "interrupted" for item in events))
            payload = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(payload["state"], "interrupted")
            self.assertFalse(payload["pid"]["active"] if isinstance(payload["pid"], dict) else False)


if __name__ == "__main__":
    unittest.main()
