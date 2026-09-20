import json
from pathlib import Path
import tempfile
import time
import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.progress_antigravity import (
    ProgressWriterAntigravity,
)


class Exp438ProgressAntigravityTests(unittest.TestCase):
    def test_background_heartbeat_updates_without_external_polling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output_antigravity"
            writer = ProgressWriterAntigravity.create(output, "a" * 64, 2, "cpu", pid=1)
            before = json.loads((output / "progress_antigravity.json").read_text())["updated_at"]
            writer.start_heartbeat(interval_seconds=0.01)
            time.sleep(0.04)
            writer.stop_heartbeat()
            after = json.loads((output / "progress_antigravity.json").read_text())["updated_at"]
            self.assertNotEqual(before, after)
    def test_create_update_and_complete_writes_required_monitoring_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output_antigravity"
            writer = ProgressWriterAntigravity.create(
                output,
                context_hash="a" * 64,
                total_datasets=1117,
                backend="cpu",
                pid=12345,
            )
            writer.update(
                stage="loading",
                current_dataset="A",
                dataset_metadata={
                    "train_count": 4,
                    "test_count": 2,
                    "train_length_histogram": {"16": 4},
                    "test_length_histogram": {"16": 2},
                    "train_value_count": 64,
                    "test_value_count": 32,
                    "train_bytes": 256,
                    "test_bytes": 128,
                },
                completed=10,
                import_count=0,
                recompute_count=10,
                success_count=9,
                abstention_count=1,
                error_count=0,
                checkpoint="A",
                per_channel_candidate_counts={
                    "high_frequency": 1,
                    "low_frequency": 2,
                    "amplitude_complexity": 3,
                },
            )
            payload = json.loads((output / "progress_antigravity.json").read_text())
            self.assertEqual(payload["pid"], 12345)
            self.assertEqual(payload["current_dataset"], "A")
            self.assertEqual(payload["dataset_metadata"]["train_count"], 4)
            self.assertEqual(payload["completed"], 10)
            self.assertGreaterEqual(payload["elapsed_seconds"], 0.0)
            self.assertIsNotNone(payload["eta_seconds"])
            writer.append_event({"event": "dataset_committed", "dataset": "A"})
            writer.mark_complete(completed=1117)
            terminal = json.loads((output / "progress_antigravity.json").read_text())
            self.assertEqual(terminal["state"], "complete")
            self.assertIsNone(terminal["current_dataset"])
            self.assertIsNone(terminal["dataset_metadata"])
            self.assertEqual((output / "run_antigravity.pid").read_text(), "12345\n")
            self.assertTrue((output / "events_antigravity.jsonl").read_text())
            self.assertTrue((output / "parent_antigravity.log").read_text())

    def test_invalid_counters_metadata_and_label_fields_fail_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output_antigravity"
            writer = ProgressWriterAntigravity.create(output, "a" * 64, 2, "cpu", pid=1)
            before = (output / "progress_antigravity.json").read_bytes()
            with self.assertRaises(ValueError):
                writer.update(stage="loading", completed=3)
            with self.assertRaisesRegex(ValueError, "forbidden prediction field"):
                writer.append_event({"event": "bad", "label": 1})
            self.assertEqual(before, (output / "progress_antigravity.json").read_bytes())

    def test_failed_progress_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output_antigravity"
            writer = ProgressWriterAntigravity.create(output, "a" * 64, 2, "cpu", pid=1)
            writer.mark_failed("synthetic_failure")
            before = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
            with self.assertRaisesRegex(ValueError, "failed progress is immutable"):
                ProgressWriterAntigravity.open_existing(output, "a" * 64, mutable=True)
            after = {path: path.read_bytes() for path in output.iterdir() if path.is_file()}
            self.assertEqual(before, after)

    def test_interrupted_progress_can_reopen_and_continue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output_antigravity"
            writer = ProgressWriterAntigravity.create(output, "a" * 64, 2, "cpu", pid=1)
            writer.update(stage="committing", current_dataset="A", completed=1)
            writer.mark_interrupted()
            reopened = ProgressWriterAntigravity.open_existing(output, "a" * 64, mutable=True)
            reopened.update(stage="idle", current_dataset=None, completed=1)
            payload = json.loads((output / "progress_antigravity.json").read_text())
            self.assertEqual(payload["state"], "running")
            self.assertEqual(payload["completed"], 1)


if __name__ == "__main__":
    unittest.main()
