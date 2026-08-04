"""Runtime progress tests for Exp425 selective recovery."""

from __future__ import annotations

import json
from pathlib import Path
import os
import copy
import pickle
import time
from collections import defaultdict
from hashlib import sha256
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp425_selective_recovery.runtime import (
    AtomicProgressWriter,
    Exp425RunCounts,
    RuntimeProgress,
    _no_model_abstention,
    _effective_record_payload,
    _HeartbeatPump,
    execute_runtime_command,
    inspect_runtime_output,
    build_exp425_runtime,
    run_selective_population,
    _compute_native,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenLaneSnapshot, FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG, Exp417FrozenConfig
from dataclasses import replace
from experiments.exp425_selective_recovery.database import Exp425DatabaseSnapshot
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery import capabilities, checkpoints
from experiments.exp425_selective_recovery.contracts import (
    Exp425RunProvenance,
    FoldIdentityCounts,
    GroupEligibilityEvidence,
    EligibilityAssessment,
    ImportedRecordEnvelope,
    NativeRecordEnvelope,
    SourceProvenGroupEvidence,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    LengthGroup,
    ProcessedGroup,
    RawSeries,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.qualification import freeze_train_contract


def _runtime_context(
    datasets: tuple[str, ...] = ("Eligible_normal_1", "Imported_normal_1", "Insufficient_normal_1"),
) -> FrozenRunContext:
    """A real tiny frozen context whose candidates remain inside the global gate."""
    config = replace(
        FROZEN_CONFIG,
        expected_datasets=len(datasets),
        expected_hard=0,
        expected_standard=0,
        expected_priority=0,
        expected_automatic_normal=0,
        database_sha256="a" * 64,
        global_workload_rate=0.5,
    )
    with patch.object(Exp417FrozenConfig, "validate", lambda _self: None):
        lanes = FrozenLaneSnapshot({}, {}, {}, {}, config=config, population=datasets)
        return FrozenRunContext(
            database_sha256="a" * 64,
            datasets=datasets,
            lanes=lanes,
            test_identity_manifest={dataset: ((0, 8), (1, 8)) for dataset in datasets},
            code_commit="c" * 40,
            runtime_versions={"python": "test"},
            config=config,
        )


def _runtime_provenance(context: FrozenRunContext, run_identity: str = "exp425-runtime-integration") -> Exp425RunProvenance:
    digest = lambda label: sha256(label.encode()).hexdigest()
    return Exp425RunProvenance(
        run_identity=run_identity,
        git_commit="c" * 40,
        database_sha256=context.database_sha256,
        lane_artifact_sha256={name: digest(name) for name in ("hard", "standard", "priority", "automatic_normal")},
        frozen_config_sha256=digest("config"),
        fold_contract_sha256=digest("folds"),
        model_contract_sha256=digest("models"),
        backend_contract_sha256=digest("backend"),
        model_seeds=(1,),
        perturbation_seeds=(2,),
        runtime_versions={"python": "test"},
        expected_datasets=context.datasets,
        predecessor_compatibility_sha256=digest("predecessor"),
    )


def _runtime_native_record(
    store: Exp425CheckpointStore,
    provenance: Exp425RunProvenance,
    dataset: str,
    *,
    status: str,
    coordinates: list[list[object]],
    reason: str,
) -> NativeRecordEnvelope:
    values = {name: sha256(f"{dataset}:{name}".encode()).hexdigest() for name in (
        "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256",
        "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256",
        "result_evidence_sha256", "backend_audit_sha256",
    )}
    trace_body = {
        "schema": "exp425.native-trace.v1",
        "run_identity": provenance.run_identity,
        "dataset": dataset,
        "run_context_sha256": store.run_context_sha256,
        **values,
        "device": "mps",
    }
    trace = sha256(checkpoints.canonical_json_bytes(trace_body)).hexdigest()
    result_payload = {
        "effective_result": {
            "status": status,
            "candidate_coordinates": coordinates,
            "result_evidence_sha256": values["result_evidence_sha256"],
        },
        "native_trace_body": trace_body,
    }
    group_evidence = {
        "eligibility_sha256": values["eligibility_sha256"],
        "processed_groups_sha256": values["processed_groups_sha256"],
    }
    model_hashes = {
        "train_identity_sha256": values["train_identity_sha256"],
        "model_evidence_sha256": values["model_evidence_sha256"],
        "frozen_train_contract_sha256": values["frozen_train_contract_sha256"],
    }
    calibration_hashes = {"scoring_evidence_sha256": values["scoring_evidence_sha256"]}
    body = {
        "schema_version": "exp425.dataset.v1", "dataset": dataset,
        "origin": "exp425_native", "run_context_sha256": store.run_context_sha256,
        "result_payload": result_payload, "group_evidence": group_evidence,
        "trace_sha256": trace, "model_hashes": model_hashes,
        "calibration_hashes": calibration_hashes,
        "backend_audit_sha256": values["backend_audit_sha256"],
        "elapsed_seconds": 0.01, "reason": reason,
    }
    return NativeRecordEnvelope(
        dataset, result_payload, group_evidence, trace_sha256=trace,
        record_payload_sha256=sha256(checkpoints.canonical_json_bytes(body)).hexdigest(),
        model_hashes=model_hashes, calibration_hashes=calibration_hashes,
        elapsed_seconds=0.01, reason=reason, **values,
    )


def _runtime_import_record(dataset: str) -> ImportedRecordEnvelope:
    evidence = {"dataset": dataset, "source": "sealed-exp424"}
    return ImportedRecordEnvelope(
        dataset,
        {"status": "scored", "candidate_coordinates": [[dataset, 0]]},
        SourceProvenGroupEvidence(evidence, sha256(checkpoints.canonical_json_bytes(evidence)).hexdigest()),
        *(sha256(f"{dataset}:{part}".encode()).hexdigest() for part in ("root", "progress", "record", "result", "attestation")),
        0.01,
        "sealed local exp424 import",
    )


def _commit_runtime_import(store: Exp425CheckpointStore, provenance: Exp425RunProvenance, record: ImportedRecordEnvelope) -> None:
    """Exercise the actual begin/stage/promote capability path for one import."""
    authority, nonce, batch_identity = object(), "source-process-nonce", "runtime-import-batch"
    attestation = record.compatibility_attestation_sha256
    stage_digest = sha256(checkpoints.canonical_json_bytes(checkpoints._import_record_body(store.run_context_sha256, record))).hexdigest()
    final_attestation = sha256(checkpoints.canonical_json_bytes({
        "batch_identity": batch_identity,
        "record_digests": {record.dataset: stage_digest},
        "attestation_sha256": attestation,
    })).hexdigest()
    capabilities._register_authority(authority, "exp424_import")
    try:
        batch = store._begin_import_batch(capabilities._mint(
            authority, kind="import_begin", run_identity=provenance.run_identity,
            batch_identity=batch_identity, attestation_sha256=attestation,
            process_nonce=nonce,
        ), batch_identity)
        store._stage_imported(capabilities._mint(
            authority, kind="import_stage", run_identity=provenance.run_identity,
            batch_identity=batch_identity, dataset=record.dataset,
            source_sha256=record.source_record_sha256, attestation_sha256=attestation,
            process_nonce=nonce, record_sha256=stage_digest,
        ), batch, record)
        store._promote_import_batch(capabilities._mint(
            authority, kind="import_promote", run_identity=provenance.run_identity,
            batch_identity=batch_identity, attestation_sha256=attestation,
            process_nonce=nonce, final_attestation_sha256=final_attestation,
        ), batch)
    finally:
        capabilities._close_authority(authority)


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

    def test_periodic_heartbeat_writes_while_stage_is_blocked_and_stops(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=1, backend="mps",
            )
            progress = replace(
                RuntimeProgress.initializing(writer.paths, "runtime-test", "new", 1, "mps"),
                state="running", stage="running", current_dataset="dataset", current_phase="preprocess",
            )
            writer.write(progress, event="dataset_start")
            pump = _HeartbeatPump(writer, progress, interval=0.01)
            pump.start()
            time.sleep(0.04)
            pump.close()
            self.assertFalse(pump._thread.is_alive())
            events = [json.loads(line) for line in writer.paths.events.read_text().splitlines()]
            heartbeats = [event for event in events if event["event"] == "heartbeat"]
            self.assertGreaterEqual(len(heartbeats), 1)
            self.assertEqual([event["heartbeat_sequence"] for event in events], sorted(event["heartbeat_sequence"] for event in events))
            self.assertTrue(all(event["current_dataset"] == "dataset" for event in heartbeats))

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
        self.assertEqual(record.group_evidence["schema"], "exp425.runtime-group-evidence.v1")
        self.assertEqual(record.model_hashes["model_evidence"]["status"], "not_applicable_oof_identity_insufficient")
        self.assertEqual(record.calibration_hashes["scoring_evidence"]["status"], "not_applicable_oof_identity_insufficient")
        self.assertEqual(trace.processed_groups_sha256, record.processed_groups_sha256)
        body = checkpoints._native_record_body(store.run_context_sha256, record)
        decoded = checkpoints._decode_record(
            checkpoints.canonical_json_bytes(body | {"record_payload_sha256": sha256(checkpoints.canonical_json_bytes(body)).hexdigest()}),
            context, native_run_identity="runtime-test", native_context_sha256=store.run_context_sha256,
        )
        self.assertIsInstance(decoded, NativeRecordEnvelope)

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

    def test_mixed_store_orchestration_commits_import_and_two_native_records_once(self) -> None:
        """Breaking origin routing, terminal accounting, or session cleanup loses durable mixed-run evidence."""
        context = _runtime_context()
        provenance = _runtime_provenance(context)
        context_decoder = patch.object(checkpoints, "_context_from_payload", return_value=context)
        context_decoder.start()
        self.addCleanup(context_decoder.stop)
        calls: dict[str, list[str]] = defaultdict(list)
        trains = {
            dataset: RawTrainDataset(dataset, (RawSeries(10, tuple(float(index) for index in range(8))),))
            for dataset in context.datasets
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, checkpoint_root = root / "output", root / "checkpoints"
            writer = AtomicProgressWriter.create(
                output_root=output, run_identity=provenance.run_identity, command="new",
                checkpoint_root=checkpoint_root, total_datasets=len(context.datasets), backend="mps",
            )
            store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
            _commit_runtime_import(store, provenance, _runtime_import_record("Imported_normal_1"))
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
            lock = os.open(root / "mps.lock", os.O_RDWR | os.O_CREAT, 0o600)

            def verify_boundary(_snapshot: Exp425DatabaseSnapshot, phase: str) -> None:
                calls["snapshot"].append(phase)

            def load_train(_snapshot: Exp425DatabaseSnapshot, dataset: str, supplied: FrozenRunContext) -> RawTrainDataset:
                self.assertIs(supplied, context)
                calls[dataset].append("TRAIN")
                return trains[dataset]

            def close_snapshot(_snapshot: Exp425DatabaseSnapshot) -> None:
                calls["snapshot"].append("close")

            records = {
                "Eligible_normal_1": _runtime_native_record(
                    store, provenance, "Eligible_normal_1", status="scored",
                    coordinates=[["Eligible_normal_1", 1]], reason="native_mps",
                ),
                "Insufficient_normal_1": _runtime_native_record(
                    store, provenance, "Insufficient_normal_1", status="abstained",
                    coordinates=[], reason="oof_identity_insufficient",
                ),
            }

            def deterministic_native(
                supplied_context: FrozenRunContext, supplied_store: Exp425CheckpointStore,
                supplied_runtime: object, dataset: str, train: RawTrainDataset,
            ) -> tuple[NativeRecordEnvelope, object]:
                self.assertIs(supplied_context, context)
                self.assertIs(supplied_store, store)
                self.assertIs(train, trains[dataset])
                calls[dataset].append("preprocess")
                if dataset == "Eligible_normal_1":
                    calls[dataset].extend(("model_qualification", "TEST", "score"))
                else:
                    calls[dataset].append("eligibility")
                return records[dataset], object()

            with patch.object(Exp425DatabaseSnapshot, "verify_boundary", verify_boundary), patch.object(
                Exp425DatabaseSnapshot, "load_train", load_train
            ), patch.object(Exp425DatabaseSnapshot, "close", close_snapshot), patch(
                "experiments.exp425_selective_recovery.runtime._require_mps"
            ), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ), patch(
                "experiments.exp425_selective_recovery.runtime._compute_native", side_effect=deterministic_native
            ), patch(
                "experiments.exp425_selective_recovery.runtime.capabilities._mint", wraps=capabilities._mint
            ) as native_mint:
                session = build_exp425_runtime(snapshot, audit, writer)
                result = run_selective_population(context, store, session)

            self.assertEqual(result.state, "complete_review_only")
            self.assertEqual(result.global_workload_status, "complete")
            self.assertEqual(result.counts, Exp425RunCounts(
                3, completed=3, imported=1, native=2, recomputed=2,
                abstained=1, oof_identity_insufficient=1, failed=0,
            ))
            self.assertEqual(calls["Imported_normal_1"], [])
            self.assertEqual(calls["Eligible_normal_1"], ["TRAIN", "preprocess", "model_qualification", "TEST", "score"])
            self.assertEqual(calls["Insufficient_normal_1"], ["TRAIN", "preprocess", "eligibility"])
            self.assertEqual(tuple(record.dataset for record in store.iter_records()), context.datasets)
            self.assertEqual(_effective_record_payload(store.load_record("Imported_normal_1"))["candidate_coordinates"], [["Imported_normal_1", 0]])
            self.assertEqual(_effective_record_payload(store.load_record("Eligible_normal_1"))["candidate_coordinates"], [["Eligible_normal_1", 1]])
            native_claims = [call.kwargs for call in native_mint.call_args_list]
            self.assertEqual(len(native_claims), 2)
            for dataset in ("Eligible_normal_1", "Insufficient_normal_1"):
                record = records[dataset]
                self.assertIn({
                    "kind": "native", "run_identity": provenance.run_identity,
                    "dataset": dataset, "context_sha256": store.run_context_sha256,
                    "trace_sha256": record.trace_sha256,
                    "record_sha256": record.record_payload_sha256,
                    "backend_sha256": record.backend_audit_sha256,
                    "process_nonce": session.process_nonce,
                }, native_claims)

            progress = json.loads(writer.paths.progress.read_text())
            events = [json.loads(line) for line in writer.paths.events.read_text().splitlines()]
            starts = [event for event in events if event["event"] == "dataset_start"]
            terminals = [event for event in events if event["event"] == "dataset_terminal"]
            self.assertEqual([event["dataset"] for event in starts], ["Eligible_normal_1", "Insufficient_normal_1"])
            self.assertEqual([event["dataset"] for event in terminals], list(context.datasets))
            self.assertEqual(progress["state"], "complete_review_only")
            self.assertIsNone(progress["current_dataset"])
            self.assertEqual(progress["completed_datasets"], 3)
            self.assertEqual(progress["failed_datasets"], 0)
            self.assertTrue(any(event["event"] == "heartbeat" for event in events))
            self.assertNotIn("label", json.dumps({"progress": progress, "events": events}).lower())
            self.assertTrue(session._closed)
            self.assertIn("close", calls["snapshot"])
            with self.assertRaises(OSError):
                os.fstat(lock)
            with self.assertRaises(ValueError):
                capabilities._mint(session, kind="native", run_identity=provenance.run_identity)
            writer.close()
            self.assertFalse(writer.paths.pid.exists())

    def test_compute_native_routes_real_eligibility_before_model_or_test_work(self) -> None:
        """Removing the identity-insufficiency branch would let TRAIN-only abstention enter model or TEST stages."""
        class ReachedEligibleRunner(RuntimeError):
            pass

        context = _runtime_context()
        provenance = _runtime_provenance(context, "exp425-routing-integration")
        dataset, insufficient_dataset = "Eligible_normal_1", "Insufficient_normal_1"
        train = RawTrainDataset(dataset, (RawSeries(10, tuple(float(index) for index in range(8))),))
        group = ProcessedGroup(
            LengthGroup(8, train.series), 0.0, 1.0, 0.0, 1.0, "candidate_eligible",
            channels={10: np.zeros((2, 8), dtype=np.float32)},
        )
        eligible_folds = tuple(FoldIdentityCounts(10, 5, 5, 5) for _ in range(5))
        insufficient_folds = tuple(FoldIdentityCounts(10, 5, 4, 5) for _ in range(5))
        eligibility = {
            dataset: GroupEligibilityEvidence(dataset, 8, "candidate_eligible", "a" * 64, "b" * 64,
                                              10, eligible_folds, 5, 5, "eligible", None),
            insufficient_dataset: GroupEligibilityEvidence(insufficient_dataset, 8, "candidate_eligible", "a" * 64, "b" * 64,
                                                           10, insufficient_folds, 5, 5,
                                                           "oof_identity_insufficient", "oof_identity_insufficient"),
        }
        routing: list[tuple[str, str]] = []
        context_decoder = patch.object(checkpoints, "_context_from_payload", return_value=context)
        context_decoder.start()
        self.addCleanup(context_decoder.stop)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root / "output", run_identity=provenance.run_identity, command="new",
                checkpoint_root=root / "checkpoints", total_datasets=3, backend="mps",
            )
            store = Exp425CheckpointStore.create(root / "checkpoints", provenance, context)
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            lock = os.open(root / "mps.lock", os.O_RDWR | os.O_CREAT, 0o600)
            audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
            abstention = _runtime_native_record(
                store, provenance, insufficient_dataset, status="abstained",
                coordinates=[], reason="oof_identity_insufficient",
            )

            def assess(name: str, groups: object, **_kwargs: object) -> EligibilityAssessment:
                routing.append((name, "eligibility"))
                return EligibilityAssessment((group,), {8: eligibility[name]})

            def reaches_runner(name: str, _context: FrozenRunContext, **_kwargs: object) -> object:
                routing.append((name, "model_test_score"))
                raise ReachedEligibleRunner(name)

            with patch.object(Exp425DatabaseSnapshot, "close"), patch(
                "experiments.exp425_selective_recovery.runtime._require_mps"
            ), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ), patch(
                "experiments.exp425_selective_recovery.runtime.preprocess_length_groups",
                side_effect=lambda _train: (group,),
            ), patch(
                "experiments.exp425_selective_recovery.runtime.apply_group_eligibility", side_effect=assess
            ), patch(
                "experiments.exp425_selective_recovery.runtime.run_dataset", side_effect=reaches_runner
            ), patch(
                "experiments.exp425_selective_recovery.runtime._no_model_abstention",
                return_value=(abstention, object()),
            ) as no_model:
                session = build_exp425_runtime(snapshot, audit, writer)
                session._fold_contract_sha256 = provenance.fold_contract_sha256
                with self.assertRaises(ReachedEligibleRunner):
                    _compute_native(context, store, session, dataset, train)
                _compute_native(context, store, session, insufficient_dataset,
                                RawTrainDataset(insufficient_dataset, train.series))
                session.close()

            self.assertEqual(routing, [(dataset, "eligibility"), (dataset, "model_test_score"),
                                       (insufficient_dataset, "eligibility")])
            self.assertEqual(no_model.call_args.args[3], insufficient_dataset)
            writer.close()

    def test_session_minted_native_commit_crashes_resume_once_at_every_durable_boundary(self) -> None:
        """Changing native intent recovery or replaying a terminal event duplicates the only resumable dataset."""
        class SimulatedProcessDeath(RuntimeError):
            pass

        context = _runtime_context(("Eligible_normal_1",))
        provenance = _runtime_provenance(context, "exp425-session-crash-resume")
        dataset = context.datasets[0]
        context_decoder = patch.object(checkpoints, "_context_from_payload", return_value=context)
        context_decoder.start()
        self.addCleanup(context_decoder.stop)
        for boundary in ("intent_durable", "record_durable", "progress_durable", "intent_removed"):
            with self.subTest(boundary=boundary), TemporaryDirectory() as directory:
                root = Path(directory)
                output, checkpoint_root = root / "output", root / "checkpoints"
                writer = AtomicProgressWriter.create(
                    output_root=output, run_identity=provenance.run_identity, command="new",
                    checkpoint_root=checkpoint_root, total_datasets=1, backend="mps",
                )
                writer.write(RuntimeProgress.initializing(
                    writer.paths, provenance.run_identity, "new", 1, "mps"
                ), event="phase_change")
                store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
                record = _runtime_native_record(
                    store, provenance, dataset, status="scored",
                    coordinates=[[dataset, 1]], reason="native_mps",
                )
                first_snapshot = object.__new__(Exp425DatabaseSnapshot)
                first_lock = os.open(root / "first-mps.lock", os.O_RDWR | os.O_CREAT, 0o600)
                audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
                with patch.object(Exp425DatabaseSnapshot, "close"), patch(
                    "experiments.exp425_selective_recovery.runtime._require_mps"
                ), patch(
                    "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=first_lock
                ):
                    first_session = build_exp425_runtime(first_snapshot, audit, writer)
                    claim = {
                        "kind": "native", "run_identity": provenance.run_identity,
                        "dataset": dataset, "context_sha256": store.run_context_sha256,
                        "trace_sha256": record.trace_sha256,
                        "record_sha256": record.record_payload_sha256,
                        "backend_sha256": record.backend_audit_sha256,
                        "process_nonce": first_session.process_nonce,
                    }
                    modified = capabilities._mint(first_session, **claim)
                    with self.assertRaises(ValueError):
                        store._commit_native(modified, replace(record, reason="mutated"))
                    self.assertEqual(list((checkpoint_root / "transactions").iterdir()), [])
                    for name, invalid in (
                        ("run_identity", "other-run"), ("dataset", "other-dataset"),
                        ("context_sha256", "0" * 64), ("trace_sha256", "1" * 64),
                        ("record_sha256", "2" * 64), ("backend_sha256", "3" * 64),
                    ):
                        wrong_claim = capabilities._mint(first_session, **(claim | {name: invalid}))
                        with self.assertRaises(ValueError, msg=name):
                            store._commit_native(wrong_claim, record)
                        with self.assertRaises(ValueError, msg=f"reused {name}"):
                            store._commit_native(wrong_claim, record)
                        self.assertEqual(list((checkpoint_root / "transactions").iterdir()), [])
                    capability = capabilities._mint(first_session, **claim)
                    with patch.object(checkpoints, "_after_native_phase", side_effect=lambda reached: (
                        (_ for _ in ()).throw(SimulatedProcessDeath(reached)) if reached == boundary else None
                    )):
                        with self.assertRaisesRegex(SimulatedProcessDeath, boundary):
                            store._commit_native(capability, record)
                    first_session.close()
                writer.close()

                resumed_store = Exp425CheckpointStore.open(checkpoint_root, provenance, context)
                self.assertEqual(
                    dict(resumed_store.completed_record_digests()),
                    {} if boundary == "intent_durable" else {dataset: record.record_payload_sha256},
                )
                resumed_writer = AtomicProgressWriter.resume(
                    output_root=output, run_identity=provenance.run_identity,
                    checkpoint_root=checkpoint_root,
                )
                second_snapshot = object.__new__(Exp425DatabaseSnapshot)
                second_lock = os.open(root / "second-mps.lock", os.O_RDWR | os.O_CREAT, 0o600)
                train = RawTrainDataset(dataset, (RawSeries(10, tuple(float(index) for index in range(8))),))
                recomputes: list[str] = []

                def load_train(_snapshot: Exp425DatabaseSnapshot, name: str, supplied: FrozenRunContext) -> RawTrainDataset:
                    self.assertEqual(name, dataset)
                    self.assertIs(supplied, context)
                    return train

                def recompute(
                    supplied_context: FrozenRunContext, supplied_store: Exp425CheckpointStore,
                    _runtime: object, name: str, supplied_train: RawTrainDataset,
                ) -> tuple[NativeRecordEnvelope, object]:
                    self.assertIs(supplied_context, context)
                    self.assertIs(supplied_store, resumed_store)
                    self.assertIs(supplied_train, train)
                    recomputes.append(name)
                    return record, object()

                with patch.object(Exp425DatabaseSnapshot, "verify_boundary"), patch.object(
                    Exp425DatabaseSnapshot, "load_train", load_train
                ), patch.object(Exp425DatabaseSnapshot, "close"), patch(
                    "experiments.exp425_selective_recovery.runtime._require_mps"
                ), patch(
                    "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=second_lock
                ), patch(
                    "experiments.exp425_selective_recovery.runtime._compute_native", side_effect=recompute
                ):
                    resumed_session = build_exp425_runtime(second_snapshot, audit, resumed_writer)
                    result = run_selective_population(context, resumed_store, resumed_session)

                self.assertEqual(recomputes, [dataset] if boundary == "intent_durable" else [])
                self.assertEqual(result.state, "complete_review_only")
                self.assertEqual(result.counts, Exp425RunCounts(
                    1, completed=1, imported=0, native=1, recomputed=1,
                    abstained=0, oof_identity_insufficient=0, failed=0,
                ))
                self.assertEqual(dict(resumed_store.completed_record_digests()), {dataset: record.record_payload_sha256})
                events = [json.loads(line) for line in resumed_writer.paths.events.read_text().splitlines()]
                terminals = [event for event in events if event["event"] == "dataset_terminal"]
                self.assertEqual([event["dataset"] for event in terminals], [dataset])
                self.assertEqual(terminals[0]["completed_datasets"], 1)
                self.assertEqual(terminals[0]["remaining_datasets"], 0)
                resumed_writer.close()
                self.assertFalse(resumed_writer.paths.pid.exists())
