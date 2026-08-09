"""Behavior tests for parent-owned Exp443 progress evidence."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)


POLICIES = (
    "linear_edge_nearest_control",
    "exclude_nonfinite_row",
    "leave_one_out_peer_median",
)
SEEDS = (20260721, 20261721)


def _unit(policy: str, dataset: str, sequence: int):
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
        WorkUnitAntigravity,
    )

    return WorkUnitAntigravity(sequence=sequence, policy=policy, dataset=dataset)


def _result(policy: str, dataset: str, *, status: str = "scored") -> PolicyDatasetResultAntigravity:
    scores = tuple(
        KernelSeedScoresAntigravity(
            kernel,
            seed,
            np.zeros(5, dtype="<f8"),
            np.zeros(2, dtype="<f8"),
            (),
        )
        for kernel in (1024, 4096)
        for seed in SEEDS
    )
    return PolicyDatasetResultAntigravity(
        policy,
        dataset,
        8,
        5,
        1,
        4,
        1,
        2,
        scores,
        {1024: (), 4096: ()},
        (),
        status,
    )


class ProgressWriterAntigravityTests(unittest.TestCase):
    def test_initial_progress_has_exact_parent_owned_schema_and_pid(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=57, pid=4242)
            payload = json.loads((root / "progress_antigravity.json").read_text())
            expected = {
                "pid", "active", "current_policy", "current_dataset", "active_datasets",
                "train_total", "train_missing", "train_used", "train_excluded", "test_rows",
                "target_length", "stage", "backend", "completed", "total", "imported",
                "recomputed", "success", "abstention", "error", "throughput", "elapsed_seconds",
                "eta_seconds", "checkpoint", "heartbeat", "latest_error",
            }
            self.assertEqual(set(payload), expected)
            self.assertEqual(payload["pid"], 4242)
            self.assertTrue(payload["active"])
            self.assertEqual(payload["total"], 57)
            self.assertIsNone(payload["eta_seconds"])
            pid = json.loads((root / "pid_antigravity.json").read_text())
            self.assertEqual(pid["pid"], 4242)
            self.assertTrue(pid["active"])
            self.assertTrue((root / "events_antigravity.jsonl").exists())
            self.assertTrue((root / "text_antigravity.log").exists())

    def test_begin_accept_and_heartbeat_keep_lowest_sequence_current(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        active = (_unit(POLICIES[0], "B", 1), _unit(POLICIES[0], "A", 0))
        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=2, pid=4242)
            writer.begin_units_antigravity(active)
            payload = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(payload["current_policy"], POLICIES[0])
            self.assertEqual(payload["current_dataset"], "A")
            self.assertEqual(payload["active_datasets"], ["A", "B"])
            writer.heartbeat_antigravity(active)
            waiting = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(waiting["stage"], "waiting_for_results")
            writer.accept_unit_antigravity(_result(POLICIES[0], "A"))
            accepted = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(accepted["completed"], 1)
            self.assertEqual(accepted["recomputed"], 1)
            self.assertEqual(accepted["success"], 1)
            self.assertEqual(accepted["current_dataset"], "B")
            self.assertEqual(accepted["train_total"], 5)
            self.assertEqual(accepted["train_missing"], 1)
            self.assertEqual(accepted["train_used"], 4)
            self.assertEqual(accepted["train_excluded"], 1)
            self.assertEqual(accepted["test_rows"], 2)
            self.assertEqual(accepted["target_length"], 8)
            events = [json.loads(line) for line in (root / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["begin_units", "heartbeat", "accept_unit"])
            self.assertIn("수락", (root / "text_antigravity.log").read_text())

    def test_accept_with_next_dimensions_writes_one_atomic_advance_without_mismatch_window(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        active = (_unit(POLICIES[0], "A", 0), _unit(POLICIES[0], "B", 1))
        next_dimensions = {
            "train_total": 99,
            "train_missing": 7,
            "train_used": 88,
            "train_excluded": 11,
            "test_rows": 12,
            "target_length": 16,
        }
        with tempfile.TemporaryDirectory(prefix="exp443_task8_progress_atomic_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=2, pid=4242)
            writer.begin_units_antigravity(active)
            with patch.object(writer, "_write_progress_antigravity", wraps=writer._write_progress_antigravity) as write:
                writer.accept_unit_and_align_antigravity(_result(POLICIES[0], "A"), next_dimensions=next_dimensions)
                self.assertEqual(write.call_count, 1)
            payload = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(payload["current_dataset"], "B")
            self.assertEqual(payload["completed"], 1)
            self.assertEqual(payload["train_total"], 99)
            self.assertEqual(payload["train_missing"], 7)
            self.assertEqual(payload["train_used"], 88)
            self.assertEqual(payload["train_excluded"], 11)
            self.assertEqual(payload["test_rows"], 12)
            self.assertEqual(payload["target_length"], 16)

    def test_small_simulated_57_unit_replay_is_complete_and_terminal_eta_is_null(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        units = tuple(
            _unit(policy, f"dataset_{index:02d}", sequence)
            for sequence, (policy, index) in enumerate(
                ( (policy, index) for policy in POLICIES for index in range(19) )
            )
        )
        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=57, pid=4242)
            writer.begin_units_antigravity(units)
            for unit in units:
                writer.accept_unit_antigravity(_result(unit.policy, unit.dataset))
            writer.terminal_antigravity("complete", None)
            payload = json.loads((root / "progress_antigravity.json").read_text())
            self.assertFalse(payload["active"])
            self.assertEqual(payload["stage"], "complete")
            self.assertEqual(payload["completed"], 57)
            self.assertEqual(payload["recomputed"], 57)
            self.assertEqual(payload["success"], 57)
            self.assertIsNone(payload["eta_seconds"])
            pid = json.loads((root / "pid_antigravity.json").read_text())
            self.assertFalse(pid["active"])
            self.assertEqual(len((root / "events_antigravity.jsonl").read_text().splitlines()), 59)

    def test_terminal_interruption_and_error_close_pid_and_preserve_latest_error(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=57, pid=4242)
            writer.terminal_antigravity("interrupted_resumable", "SIGTERM")
            payload = json.loads((root / "progress_antigravity.json").read_text())
            self.assertFalse(payload["active"])
            self.assertEqual(payload["stage"], "interrupted_resumable")
            self.assertEqual(payload["latest_error"], "SIGTERM")
            self.assertIsNone(payload["eta_seconds"])
            pid = json.loads((root / "pid_antigravity.json").read_text())
            self.assertFalse(pid["active"])

    def test_resume_reopens_interrupted_logs_and_restores_elapsed_counters(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        units = (_unit(POLICIES[0], "A", 0), _unit(POLICIES[0], "B", 1))
        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_resume_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=2, pid=4242)
            writer.begin_units_antigravity(units)
            writer.accept_unit_antigravity(_result(POLICIES[0], "A"))
            writer.terminal_antigravity("interrupted_resumable", "SIGTERM")
            before = json.loads((root / "progress_antigravity.json").read_text())
            events_before = (root / "events_antigravity.jsonl").read_bytes()
            text_before = (root / "text_antigravity.log").read_bytes()

            resumed = ProgressWriterAntigravity.open_for_resume_antigravity(
                root,
                total=2,
                pid=5252,
            )
            self.assertEqual(resumed.completed_count_antigravity, 1)
            reopened = json.loads((root / "progress_antigravity.json").read_text())
            self.assertTrue(reopened["active"])
            self.assertEqual(reopened["completed"], before["completed"])
            self.assertEqual(reopened["recomputed"], before["recomputed"])
            self.assertEqual(reopened["success"], before["success"])
            self.assertGreaterEqual(reopened["elapsed_seconds"], before["elapsed_seconds"])
            self.assertTrue((root / "events_antigravity.jsonl").read_bytes().startswith(events_before))
            self.assertTrue((root / "text_antigravity.log").read_bytes().startswith(text_before))
            pid = json.loads((root / "pid_antigravity.json").read_text())
            self.assertEqual(pid["pid"], 5252)
            self.assertTrue(pid["active"])

            resumed.begin_units_antigravity((units[1],))
            resumed.accept_unit_antigravity(_result(POLICIES[0], "B"))
            resumed.terminal_antigravity("complete", None)
            finished = json.loads((root / "progress_antigravity.json").read_text())
            self.assertEqual(finished["completed"], 2)
            self.assertEqual(finished["stage"], "complete")

    def test_heartbeat_after_terminal_rejects_without_reactivating_pid_or_logs(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import (
            ProgressWriterAntigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp443_task7_progress_terminal_heartbeat_antigravity_") as directory:
            root = Path(directory) / "attempt_antigravity"
            root.mkdir()
            writer = ProgressWriterAntigravity(root, total=2, pid=4242)
            writer.terminal_antigravity("complete", None)
            progress_before = (root / "progress_antigravity.json").read_bytes()
            events_before = (root / "events_antigravity.jsonl").read_bytes()
            text_before = (root / "text_antigravity.log").read_bytes()
            pid_before = json.loads((root / "pid_antigravity.json").read_text())
            with self.assertRaisesRegex(ValueError, "terminal"):
                writer.heartbeat_antigravity(())
            self.assertEqual((root / "progress_antigravity.json").read_bytes(), progress_before)
            self.assertEqual((root / "events_antigravity.jsonl").read_bytes(), events_before)
            self.assertEqual((root / "text_antigravity.log").read_bytes(), text_before)
            self.assertEqual(json.loads((root / "pid_antigravity.json").read_text()), pid_before)
            self.assertFalse(pid_before["active"])


if __name__ == "__main__":
    unittest.main()
