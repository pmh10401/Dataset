"""Runtime progress tests for Exp425 selective recovery."""

from __future__ import annotations

import json
from pathlib import Path
import os
import copy
import pickle
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp425_selective_recovery.runtime import (
    AtomicProgressWriter,
    RuntimeProgress,
    _no_model_abstention,
    _effective_record_payload,
    execute_runtime_command,
    inspect_runtime_output,
    build_exp425_runtime,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenLaneSnapshot, FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG, Exp417FrozenConfig
from dataclasses import replace
from experiments.exp425_selective_recovery.database import Exp425DatabaseSnapshot
from experiments.exp425_selective_recovery.contracts import FoldIdentityCounts, GroupEligibilityEvidence
from experiments.exp417_causal_multihorizon_shadow.contracts import LengthGroup, ProcessedGroup, RawSeries
from experiments.exp417_causal_multihorizon_shadow.qualification import freeze_train_contract


class AtomicProgressWriterTests(unittest.TestCase):
    def test_small_frozen_context_fixture_uses_real_three_dataset_universe(self) -> None:
        config = replace(
            FROZEN_CONFIG, expected_datasets=3, expected_hard=1, expected_standard=1,
            expected_priority=1, expected_automatic_normal=0, database_sha256="a" * 64,
        )
        datasets = ("Eligible_normal_1", "Imported_normal_1", "Insufficient_normal_1")
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None):
            lanes = FrozenLaneSnapshot(
                hard={"Eligible_normal_1": (0,)}, standard={"Imported_normal_1": (0,)},
                priority={"Insufficient_normal_1": (0,)}, automatic_normal={}, config=config,
                population=datasets,
            )
            context = FrozenRunContext(
                database_sha256="a" * 64, datasets=datasets, lanes=lanes,
                test_identity_manifest={dataset: ((0, 8),) for dataset in datasets},
                code_commit="c" * 40, runtime_versions={"python": "test"}, config=config,
            )
        self.assertEqual(context.datasets, datasets)
        self.assertEqual(context.test_sizes["Eligible_normal_1"], 1)
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

    def test_invalid_resume_does_not_claim_pid_or_mutate_output(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=2, backend="mps",
            )
            state = RuntimeProgress.initializing(writer.paths, "runtime-test", "new", 2, "mps")
            writer.write(state, event="phase_change")
            writer.close()
            writer._write_pid()
            writer.paths.events.write_bytes(writer.paths.events.read_bytes()[:-1])
            before = {path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in root.iterdir()}
            with self.assertRaisesRegex(ValueError, "truncated"):
                AtomicProgressWriter.resume(output_root=root, run_identity="runtime-test", checkpoint_root=root / "checkpoints")
            after = {path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in root.iterdir()}
            self.assertEqual(before, after)

    def test_inactive_resume_accepts_absent_pid_but_rejects_live_owner(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=1, backend="mps",
            )
            state = RuntimeProgress.initializing(writer.paths, "runtime-test", "new", 1, "mps")
            writer.write(state, event="phase_change")
            writer.close()
            self.assertFalse(writer.paths.pid.exists())
            resumed = AtomicProgressWriter.resume(
                output_root=root, run_identity="runtime-test", checkpoint_root=root / "checkpoints",
                claim_pid=False,
            )
            resumed.claim_pid()
            with self.assertRaisesRegex(ValueError, "live process"):
                AtomicProgressWriter.resume(
                    output_root=root, run_identity="runtime-test", checkpoint_root=root / "checkpoints",
                    claim_pid=True,
                )
            resumed.close()

    def test_close_never_removes_a_replaced_pid_lease(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=1, backend="mps",
            )
            replacement = json.dumps(
                {"schema": "exp425.runtime-pid.v1", "run_identity": "runtime-test", "pid": os.getpid(), "nonce": "new-owner"},
                sort_keys=True, separators=(",", ":"),
            ).encode()
            writer.paths.pid.write_bytes(replacement)
            writer.close()
            self.assertEqual(writer.paths.pid.read_bytes(), replacement)

    def test_no_model_oof_abstention_has_empty_decisions_and_explicit_na_evidence(self) -> None:
        dataset = "Insufficient_normal_1"
        config = replace(FROZEN_CONFIG, expected_datasets=1, expected_hard=0, expected_standard=0,
                         expected_priority=0, expected_automatic_normal=0, database_sha256="a" * 64)
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None):
            lanes = FrozenLaneSnapshot({}, {}, {}, {}, config=config, population=(dataset,))
            context = FrozenRunContext("a" * 64, (dataset,), lanes, {dataset: ((0, 8),)}, "c" * 40, {"python": "test"}, config)
        group = ProcessedGroup(LengthGroup(8, (RawSeries(0, tuple(float(i) for i in range(8))),)),
                               0.0, 1.0, 0.0, 1.0, "insufficient_train_abstain")
        folds = tuple(FoldIdentityCounts(5, 1, 4, 1) for _ in range(5))
        eligibility = GroupEligibilityEvidence(dataset, 8, "candidate_eligible", "a" * 64,
                                               "b" * 64, 1, folds, 4, 1,
                                               "oof_identity_insufficient", "oof_identity_insufficient")
        audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
        store = SimpleNamespace(provenance=SimpleNamespace(run_identity="runtime-test"), run_context_sha256="d" * 64)
        runtime = SimpleNamespace(
            audit=audit,
            _eligibility={dataset: {8: eligibility}},
            runner=SimpleNamespace(freeze=lambda name, qualification, models: freeze_train_contract(name, qualification.groups, qualification.backend, train_identity_sha256=models.train_identity_sha256)),
        )
        record, trace = _no_model_abstention(context, store, runtime, dataset, (group,), 0.25)
        effective = record.result_payload["effective_result"]
        self.assertEqual(effective["status"], "abstained")
        self.assertEqual(record.reason, "oof_identity_insufficient")
        self.assertEqual(record.model_hashes["model_evidence"]["status"], "not_applicable")
        self.assertEqual(record.calibration_hashes["scoring_evidence"]["status"], "not_applicable")
        self.assertEqual(trace.processed_groups_sha256, record.processed_groups_sha256)

    def test_effective_record_decoder_handles_native_and_imported_envelopes(self) -> None:
        native = SimpleNamespace(origin="exp425_native", result_payload={
            "effective_result": {"status": "scored", "candidate_coordinates": [["n", 1]]},
        })
        imported = SimpleNamespace(origin="exp424_import", result_payload={
            "status": "abstained", "candidate_coordinates": [],
        })
        self.assertEqual(_effective_record_payload(native)["status"], "scored")
        self.assertEqual(_effective_record_payload(imported)["status"], "abstained")

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
            with patch.object(Exp425DatabaseSnapshot, "close"):
                session.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                session._require_live()
            with self.assertRaisesRegex(ValueError, "MPS audit"):
                build_exp425_runtime(snapshot, BackendAudit("cpu", True, 0.0, 1.0, 1.1, False), writer)

    def test_runtime_session_rejects_public_and_private_binding_reassignment(self) -> None:
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
            for name, value in (("snapshot", snapshot), ("audit", audit), ("writer", writer), ("runner", session.runner), ("_snapshot", snapshot), ("_runner", session.runner)):
                with self.assertRaises(AttributeError, msg=name):
                    setattr(session, name, value)
            with patch.object(Exp425DatabaseSnapshot, "close"):
                session.close()
