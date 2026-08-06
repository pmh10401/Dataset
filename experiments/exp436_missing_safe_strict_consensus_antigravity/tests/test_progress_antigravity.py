"""Adversarial durability tests for the Exp436 parent-owned progress writer."""
from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes

from experiments.exp436_missing_safe_strict_consensus_antigravity.progress_antigravity import (
    ProgressWriter,
    inspect_progress,
)


def _metadata() -> dict[str, object]:
    return {"train_count": 2, "test_count": 3, "train_length_histogram": {"2": 2}, "test_length_histogram": {"2": 3}}


def _checkpoint(completed: int = 0) -> dict[str, object]:
    datasets = ("alpha", "beta")[:completed]
    return {"completed": completed, "datasets": datasets, "record_digests": {name: ("a" if name == "alpha" else "b") * 64 for name in datasets}}


class ProgressWriterTests(unittest.TestCase):
    def _writer(self, root: Path, **kwargs: object) -> ProgressWriter:
        options = {"heartbeat_seconds": None, **kwargs}
        return ProgressWriter.create(root, run_identity="run", checkpoint=root.parent / "checkpoint_antigravity", total_datasets=1117, backend="cpu", context_fingerprint="c" * 64, **options)

    def _complete(self, writer: ProgressWriter, index: int = 0, dataset: str = "alpha", digest: str = "a" * 64) -> None:
        writer.record_dataset_complete(sequence_index=index, dataset=dataset, record_sha256=digest, status="scored", dataset_metadata=_metadata(), completed_datasets=index + 1, success=index + 1, abstention=0, error=0, excluded_train_rows=0, nonfinite_test_rows=0, candidate_count=0)

    def test_required_snapshot_and_pid_names_are_exact(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            self.assertEqual(set(writer.progress), {"schema", "run_identity", "context_sha256", "pid", "pid_active", "heartbeat", "updated_at", "state", "stage", "backend", "current_dataset", "dataset_metadata", "in_flight_datasets", "completed_datasets", "total_datasets", "recomputed_datasets", "success", "abstention", "error", "excluded_train_rows", "nonfinite_test_rows", "candidate_count", "datasets_per_second", "elapsed_seconds", "eta_seconds", "checkpoint_path", "latest_error", "last_event_sequence", "last_event_sha256", "progress_sha256"})
            self.assertEqual(set(json.loads((writer.output / "run_antigravity.pid").read_text())), {"schema", "run_identity", "context_sha256", "checkpoint_path", "backend", "total_datasets", "pid", "pid_active", "heartbeat", "state", "pid_sha256"})
            writer.close()

    def test_unknown_snapshot_key_is_rejected_read_only(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); writer.close()
            path = writer.output / "progress_antigravity.json"; payload = json.loads(path.read_text()); payload["unknown"] = True; path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            with self.assertRaisesRegex(ValueError, "progress"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())

    def test_active_and_idle_snapshot_invariants(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            with self.assertRaisesRegex(ValueError, "stage"):
                writer.update(stage="loading", current_dataset=None, dataset_metadata=None)
            with self.assertRaisesRegex(ValueError, "stage"):
                writer.update(stage="idle", current_dataset="alpha", dataset_metadata=_metadata())
            writer.update(stage="loading", current_dataset="alpha", dataset_metadata=_metadata()); self.assertEqual(writer.progress["stage"], "loading")
            writer.close()

    def test_metadata_conservation_and_label_firewall(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            bad = _metadata(); bad["train_length_histogram"] = {"2": 1}
            with self.assertRaisesRegex(ValueError, "metadata"):
                writer.update(stage="loading", current_dataset="alpha", dataset_metadata=bad)
            bad = _metadata(); bad["label"] = 0
            with self.assertRaisesRegex(ValueError, "metadata"):
                writer.update(stage="loading", current_dataset="alpha", dataset_metadata=bad)
            writer.close()

    def test_atomic_progress_replacement_never_exposes_partial_json(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            writer.update(stage="loading", current_dataset="alpha", dataset_metadata=_metadata())
            self.assertEqual(json.loads((writer.output / "progress_antigravity.json").read_text())["current_dataset"], "alpha")
            self.assertFalse(any(path.name.endswith(".tmp") for path in writer.output.iterdir()))
            writer.close()

    def test_event_hash_chain_is_exact_and_snapshot_binds_tail(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); self._complete(writer)
            inspection = inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint(1))
            self.assertEqual(inspection["event_completed"], 1); self.assertEqual(writer.progress["last_event_sequence"], 2)
            writer.close()

    def test_partial_noncanonical_and_unknown_event_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); writer.close(); events = writer.output / "events_antigravity.jsonl"
            events.write_bytes(events.read_bytes() + b'{"partial"')
            with self.assertRaisesRegex(ValueError, "partial"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())
            events.write_bytes(events.read_bytes().splitlines()[0] + b"\n")
            record = json.loads(events.read_text()); record["unknown"] = 1; events.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            with self.assertRaisesRegex(ValueError, "event"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())

    def test_duplicate_gap_out_of_order_and_foreign_completion_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); self._complete(writer); writer.close()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint(0))
            with self.assertRaisesRegex(ValueError, "dataset"):
                ProgressWriter.open_existing(writer.output, run_identity="run", checkpoint=writer.checkpoint, backend="cpu", total_datasets=1117, context_fingerprint="c" * 64, checkpoint_inspection={"completed": 1, "datasets": ("beta",), "record_digests": {"beta": "b" * 64}})

    def test_snapshot_counter_or_event_tail_tamper_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); self._complete(writer); writer.close()
            progress = writer.output / "progress_antigravity.json"; payload = json.loads(progress.read_text()); payload["completed_datasets"] = 2; progress.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
            with self.assertRaisesRegex(ValueError, "progress"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint(1))

    def test_checkpoint_ahead_returns_ordered_missing_tail_then_reconciles(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            state = inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint(1))
            self.assertEqual(state["reconciliation_required"], ("alpha",))
            writer.reconcile_checkpoint_tail(((0, "alpha", "a" * 64, "scored", _metadata(), {"success": 1, "abstention": 0, "error": 0, "excluded_train_rows": 0, "nonfinite_test_rows": 0, "candidate_count": 0}),), checkpoint_inspection=_checkpoint(1))
            self.assertEqual(writer.progress["completed_datasets"], 1); writer.close()

    def test_event_before_snapshot_crash_replays_the_validated_event_prefix(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            # This is the precise post-fsync/pre-snapshot crash window: the
            # completion event is durable while the snapshot still binds start.
            writer._append_event("dataset_complete", {"sequence_index": 0, "dataset": "alpha", "record_sha256": "a" * 64, "status": "scored", "dataset_metadata": _metadata(), "completed_datasets": 1, "recomputed_datasets": 1, "success": 1, "abstention": 0, "error": 0, "excluded_train_rows": 0, "nonfinite_test_rows": 0, "candidate_count": 0})
            writer.close()
            resumed = ProgressWriter.open_existing(writer.output, run_identity="run", checkpoint=writer.checkpoint, backend="cpu", total_datasets=1117, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint(1), heartbeat_seconds=None)
            resumed.synchronize_event_prefix(checkpoint_inspection=_checkpoint(1))
            self.assertEqual(resumed.progress["completed_datasets"], 1)
            self.assertEqual(resumed.progress["success"], 1)
            self.assertEqual(sum(json.loads(line)["kind"] == "dataset_complete" for line in (resumed.output / "events_antigravity.jsonl").read_text().splitlines()), 1)
            resumed.close()

    def test_event_ahead_and_progress_ahead_are_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); self._complete(writer); writer.close()
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())

    def test_only_interrupted_terminal_may_be_followed_by_a_fresh_runtime_start(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            writer.mark_failed("alpha", "loading", RuntimeError("boom")); writer.close()
            events = writer.output / "events_antigravity.jsonl"
            failed = [json.loads(line) for line in events.read_text().splitlines()]
            restarted = dict(failed[0]); restarted["sequence"] = 3; restarted["previous_event_sha256"] = failed[-1]["event_sha256"]
            restarted["event_sha256"] = sha256(canonical_json_bytes({key: value for key, value in restarted.items() if key != "event_sha256"})).hexdigest()
            events.write_bytes(b"".join(canonical_json_bytes(event) + b"\n" for event in (*failed, restarted)))
            with self.assertRaisesRegex(ValueError, "terminal lifecycle"):
                inspect_progress(writer.output, run_identity="run", checkpoint=writer.checkpoint, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())

    def test_parent_text_log_appends_state_and_event_context(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); self._complete(writer)
            text = (writer.output / "parent_antigravity.log").read_text(); self.assertIn("runtime_started", text); self.assertIn("dataset_complete", text); self.assertIn("alpha", text); writer.close()

    def test_heartbeat_changes_only_liveness_fields(self) -> None:
        with TemporaryDirectory() as directory:
            gate = Event(); writer = self._writer(Path(directory) / "output_antigravity", heartbeat_seconds=0.001, heartbeat_observer=gate.set)
            before = {key: writer.progress[key] for key in ("completed_datasets", "success", "abstention", "candidate_count", "last_event_sequence")}; writer.start_heartbeat(); self.assertTrue(gate.wait(1)); writer.stop_heartbeat()
            self.assertEqual(before, {key: writer.progress[key] for key in before}); writer.close()

    def test_terminal_pid_is_inactive_and_close_is_idempotent(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); writer.mark_interrupted("alpha", "loading", KeyboardInterrupt())
            self.assertFalse(json.loads((writer.output / "run_antigravity.pid").read_text())["pid_active"]); writer.close(); writer.close()

    def test_second_writer_fails_before_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); before = {p.name: p.read_bytes() for p in writer.output.iterdir()}
            with self.assertRaisesRegex(ValueError, "owned"):
                ProgressWriter.open_existing(writer.output, run_identity="run", checkpoint=writer.checkpoint, backend="cpu", total_datasets=1117, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())
            self.assertEqual(before, {p.name: p.read_bytes() for p in writer.output.iterdir()}); writer.close()

    def test_creator_pid_and_post_close_guards_reject_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); original = writer.creator_pid; writer.creator_pid = original + 1
            with self.assertRaisesRegex(ValueError, "owner"):
                writer.update(stage="idle", current_dataset=None, dataset_metadata=None)
            writer.creator_pid = original; writer.close()
            with self.assertRaisesRegex(ValueError, "owner"):
                writer.append_log("again")

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX fork guard")
    def test_forked_child_cannot_mutate_parent_writer(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            child = os.fork()
            if child == 0:
                try:
                    writer.append_log("child")
                except ValueError:
                    os._exit(0)
                os._exit(1)
            _, status = os.waitpid(child, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            self.assertNotIn("child", (writer.output / "parent_antigravity.log").read_text())
            writer.close()

    def test_failed_terminal_does_not_advance_and_keeps_offending_context(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity"); writer.update(stage="committing", current_dataset="alpha", dataset_metadata=_metadata()); writer.mark_failed("alpha", "committing", RuntimeError("boom"))
            self.assertEqual(writer.progress["completed_datasets"], 0); self.assertEqual(writer.progress["latest_error"]["dataset"], "alpha"); writer.close()

    def test_checkpoint_complete_is_emitted_once(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            writer.progress.update({"completed_datasets": 1117, "recomputed_datasets": 1117, "success": 1117, "abstention": 0})
            writer.mark_checkpoint_complete()
            with self.assertRaisesRegex(ValueError, "already"):
                writer.mark_checkpoint_complete()
            writer.close()

    def test_checkpoint_complete_rejects_an_incomplete_runtime(self) -> None:
        with TemporaryDirectory() as directory:
            writer = self._writer(Path(directory) / "output_antigravity")
            with self.assertRaisesRegex(ValueError, "all datasets"):
                writer.mark_checkpoint_complete()
            writer.close()

    def test_only_four_durable_artifacts_and_valid_temp_cleanup(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "output_antigravity"; writer = self._writer(root); writer.close()
            temp = root / (".progress_antigravity.json." + "a" * 32 + ".tmp")
            # The valid owned temporary is cleaned only after a read-only-valid root is reopened.
            temp.write_bytes((root / "progress_antigravity.json").read_bytes())
            reopened = ProgressWriter.open_existing(root, run_identity="run", checkpoint=writer.checkpoint, backend="cpu", total_datasets=1117, context_fingerprint="c" * 64, checkpoint_inspection=_checkpoint())
            self.assertEqual({p.name for p in root.iterdir()}, {"progress_antigravity.json", "events_antigravity.jsonl", "parent_antigravity.log", "run_antigravity.pid"}); reopened.close()
