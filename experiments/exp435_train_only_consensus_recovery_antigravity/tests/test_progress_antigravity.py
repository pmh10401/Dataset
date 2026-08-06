"""TDD checks for Exp435 progress, events, and PID lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.progress_antigravity import (
    ProgressWriter,
    _validate_output_inventory,
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
            self.assertTrue(payload["checkpoint"].endswith("/checkpoint"))
            self.assertEqual(payload["context_fingerprint"], "unbound")
            writer.close()
            pid = json.loads((root / "run_antigravity.pid").read_text())
            self.assertFalse(pid["active"])
            self.assertEqual(pid["backend"], "cpu")
            self.assertEqual(pid["total_datasets"], 2)
            events = [json.loads(line) for line in (root / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertTrue(all(item["context_fingerprint"] == "unbound" for item in events))

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

    def test_resume_removes_only_internal_atomic_temp_and_rejects_unknown_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            writer = ProgressWriter.create(
                root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                total_datasets=1, backend="cpu", heartbeat_seconds=None,
            )
            writer.close()
            internal = root / (".progress_antigravity.json." + "a" * 32 + ".tmp")
            internal.write_bytes(b"partial")
            resumed = ProgressWriter.open_existing(
                root,
                run_identity="exp435-fixture",
                checkpoint=Path(directory) / "cp",
                process_is_live=lambda pid: False,
            )
            resumed.close()
            self.assertFalse(internal.exists())
            unknown = root / ".progress_antigravity.json.unknown.tmp"
            unknown.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "inventory"):
                ProgressWriter.open_existing(
                    root,
                    run_identity="exp435-fixture",
                    checkpoint=Path(directory) / "cp",
                    process_is_live=lambda pid: False,
                )

    def test_pid_atomic_temp_is_reconciled_and_read_only_validation_does_not_mutate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            writer = ProgressWriter.create(
                root, run_identity="exp435-fixture", checkpoint=Path(directory) / "cp",
                total_datasets=1, backend="cpu", heartbeat_seconds=None,
            )
            writer.close()
            internal = root / (".run_antigravity.pid." + "b" * 32 + ".tmp")
            internal.write_bytes(b"partial")
            before = internal.read_bytes()
            with self.assertRaisesRegex(ValueError, "pending atomic temporary"):
                _validate_output_inventory(root, mutable=False)
            self.assertEqual(internal.read_bytes(), before)
            resumed = ProgressWriter.open_existing(
                root,
                run_identity="exp435-fixture",
                checkpoint=Path(directory) / "cp",
                process_is_live=lambda pid: False,
            )
            resumed.close()
            self.assertFalse(internal.exists())
            unknown = root / (".run_antigravity.pid." + "c" * 31 + ".tmp")
            unknown.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "inventory"):
                ProgressWriter.open_existing(
                    root,
                    run_identity="exp435-fixture",
                    checkpoint=Path(directory) / "cp",
                    process_is_live=lambda pid: False,
                )

    def test_output_binding_rejects_checkpoint_backend_total_and_fingerprint_swaps(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            checkpoint = Path(directory) / "checkpoint-cpu"
            writer = ProgressWriter.create(
                root,
                run_identity="exp435-fixture",
                checkpoint=checkpoint,
                total_datasets=1,
                backend="cpu",
                context_fingerprint="a" * 64,
                heartbeat_seconds=None,
            )
            writer.close()
            with self.assertRaisesRegex(ValueError, "binding"):
                ProgressWriter.open_existing(
                    root,
                    run_identity="exp435-fixture",
                    checkpoint=Path(directory) / "checkpoint-mlx",
                    backend="mlx",
                    total_datasets=1,
                    context_fingerprint="b" * 64,
                    process_is_live=lambda pid: False,
                )
            progress_path = root / "progress_antigravity.json"
            progress = json.loads(progress_path.read_text())
            progress["context_fingerprint"] = "c" * 64
            progress_path.write_bytes(
                json.dumps(progress, sort_keys=True, separators=(",", ":")).encode()
            )
            with self.assertRaisesRegex(ValueError, "binding"):
                ProgressWriter.open_existing(
                    root,
                    run_identity="exp435-fixture",
                    checkpoint=checkpoint,
                    backend="cpu",
                    total_datasets=1,
                    context_fingerprint="a" * 64,
                    process_is_live=lambda pid: False,
                )
            events_root = Path(directory) / "events-output"
            events_writer = ProgressWriter.create(
                events_root,
                run_identity="exp435-fixture",
                checkpoint=checkpoint,
                total_datasets=1,
                backend="cpu",
                context_fingerprint="a" * 64,
                heartbeat_seconds=None,
            )
            events_writer.close()
            event_path = events_root / "events_antigravity.jsonl"
            event = json.loads(event_path.read_text().splitlines()[0])
            event["backend"] = "mlx"
            event_path.write_bytes((json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode())
            with self.assertRaisesRegex(ValueError, "binding"):
                ProgressWriter.open_existing(
                    events_root,
                    run_identity="exp435-fixture",
                    checkpoint=checkpoint,
                    backend="cpu",
                    total_datasets=1,
                    context_fingerprint="a" * 64,
                    process_is_live=lambda pid: False,
                )


if __name__ == "__main__":
    unittest.main()
