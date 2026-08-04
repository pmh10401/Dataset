"""Runtime progress tests for Exp425 selective recovery."""

from __future__ import annotations

import json
from pathlib import Path
import os
import copy
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp425_selective_recovery.runtime import (
    AtomicProgressWriter,
    RuntimeProgress,
    execute_runtime_command,
    inspect_runtime_output,
    build_exp425_runtime,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp425_selective_recovery.database import Exp425DatabaseSnapshot


class AtomicProgressWriterTests(unittest.TestCase):
    def test_write_is_atomic_and_events_are_canonical_and_label_free(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root,
                run_identity="runtime-test",
                command="new",
                checkpoint_root=root / "checkpoints",
                total_datasets=3,
                backend="mps",
            )
            state = RuntimeProgress.initializing(writer.paths, "runtime-test", "new", 3, "mps")
            writer.write(state, event="phase_change")
            payload = json.loads(writer.paths.progress.read_bytes())
            self.assertEqual(payload["state"], "initializing")
            self.assertEqual(payload["completed_datasets"], 0)
            self.assertIsNone(payload["current_dataset"])
            self.assertFalse(any(path.name.startswith(".progress.json.") for path in root.iterdir()))
            events = writer.paths.events.read_bytes().splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0], json.dumps(json.loads(events[0]), sort_keys=True, separators=(",", ":")).encode())
            self.assertNotIn(b"label", writer.paths.progress.read_bytes().lower())
            self.assertNotIn(b"target", writer.paths.events.read_bytes().lower())

    def test_new_and_resume_reject_their_wrong_root_lifecycle_before_launch(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                execute_runtime_command(
                    "new", output_root=existing, checkpoint_root=root / "checkpoint",
                    provenance=None, context=None, snapshot=None, audit=None,
                )
            with self.assertRaisesRegex(ValueError, "is absent"):
                execute_runtime_command(
                    "resume", output_root=root / "output", checkpoint_root=root / "checkpoint",
                    provenance=None, context=None, snapshot=None, audit=None,
                )

    def test_inspection_is_read_only_and_resume_samples_reject_duplicate_datasets(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=2, backend="mps",
            )
            state = RuntimeProgress.initializing(writer.paths, "runtime-test", "new", 2, "mps")
            writer.write(state, event="phase_change")
            before = {path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in root.iterdir()}
            inspected = inspect_runtime_output(root, expected_run_identity="runtime-test")
            self.assertEqual(inspected["state"], "initializing")
            after = {path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in root.iterdir()}
            self.assertEqual(before, after)
            event = {
                "schema": "exp425.runtime-event.v1", "event": "dataset_terminal",
                "timestamp": 1.0, "run_identity": "runtime-test", "pid": os.getpid(),
                "state": "running", "stage": "running", "current_dataset": None,
                "dataset": "same", "elapsed_seconds": 1.0,
            }
            line = json.dumps(event, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            writer.paths.events.write_bytes(line + line)
            with self.assertRaisesRegex(ValueError, "terminal event"):
                writer.read_terminal_samples()

    def test_runtime_session_is_opaque_and_requires_a_passing_mps_audit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=1, backend="mps",
            )
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
            lock = os.open(root / "lock", os.O_RDWR | os.O_CREAT, 0o600)
            with patch("experiments.exp425_selective_recovery.runtime._require_mps"), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ):
                session = build_exp425_runtime(snapshot, audit, writer)
            with self.assertRaises(TypeError):
                copy.copy(session)
            with self.assertRaises(TypeError):
                pickle.dumps(session)
            session.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                session._require_live()
            with self.assertRaisesRegex(ValueError, "MPS audit"):
                build_exp425_runtime(snapshot, BackendAudit("cpu", True, 0.0, 1.0, 1.1, False), writer)
