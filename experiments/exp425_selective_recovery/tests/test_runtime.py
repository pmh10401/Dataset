"""Runtime progress tests for Exp425 selective recovery."""

from __future__ import annotations

import json
import inspect
from pathlib import Path
import os
import copy
import pickle
import time
from collections import defaultdict
from hashlib import sha256
from types import MappingProxyType, SimpleNamespace
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
    _apply_compact_global_workload,
    _HeartbeatPump,
    execute_runtime_command,
    inspect_runtime_output,
    build_exp425_runtime,
    Exp425RuntimeSession,
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
    MPSExecutionEvidence,
    NativeRecordEnvelope,
    SourceProvenGroupEvidence,
)
from experiments.exp425_selective_recovery.contracts import MPSAdoptionAssessment


def _mps_assessment() -> MPSAdoptionAssessment:
    return MPSAdoptionAssessment("/Users/mac/Documents/Dataset/exp417_causal_multihorizon_shadow/task7_backend_audit.json", "f77efa941f6b33fe6687b8ac0c2b01c152dac9c15e8b0966e3d9bb41b119b592", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, 1.1, False, "speedup_below_threshold")


def _runtime_mps_execution(provenance: Exp425RunProvenance) -> MPSExecutionEvidence:
    return MPSExecutionEvidence(
        True, "mps", "mps", True, True, False, {"final:8:1": "mps"},
        "2.0.0", "4" * 40, {"runner": "b" * 64}, provenance.git_commit,
        "synthetic timer", 0,
    )
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    LengthGroup,
    ProcessedGroup,
    RawSeries,
    RawTrainDataset,
    SeedDecision,
    ShadowCandidate,
    TestInstanceDecision,
)
from experiments.exp417_causal_multihorizon_shadow import contracts as exp417_contracts
from experiments.exp417_causal_multihorizon_shadow import qualification as exp417_qualification
from experiments.exp417_causal_multihorizon_shadow.qualification import apply_global_workload, freeze_train_contract


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
        "schema": "exp425.native-trace.v2",
        "run_identity": provenance.run_identity,
        "dataset": dataset,
        "run_context_sha256": store.run_context_sha256,
        **values,
        "device": "mps",
    }
    result_payload = {
        "effective_result": {
            "status": status,
            "candidate_coordinates": coordinates,
            "result_evidence_sha256": values["result_evidence_sha256"],
            "result_source_sha256": sha256(checkpoints.canonical_json_bytes({
                "status": status, "candidate_coordinates": coordinates,
            })).hexdigest(),
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
    adoption, execution = _mps_assessment(), _runtime_mps_execution(provenance)
    trace_body.update({"mps_adoption_sha256": adoption.sha256, "mps_execution_sha256": execution.sha256})
    trace = sha256(checkpoints.canonical_json_bytes(trace_body)).hexdigest()
    provisional = NativeRecordEnvelope(
        dataset, result_payload, group_evidence, trace_sha256=trace,
        record_payload_sha256="0" * 64, model_hashes=model_hashes,
        calibration_hashes=calibration_hashes, elapsed_seconds=0.01, reason=reason,
        mps_adoption=adoption, mps_execution=execution, **values,
    )
    return replace(provisional, record_payload_sha256=sha256(checkpoints.canonical_json_bytes(
        checkpoints._native_record_body(store.run_context_sha256, provisional)
    )).hexdigest())


def _runtime_import_record(dataset: str) -> ImportedRecordEnvelope:
    evidence = {"dataset": dataset, "source": "sealed-exp424"}
    source_result = sha256(f"{dataset}:result".encode()).hexdigest()
    return ImportedRecordEnvelope(
        dataset,
        {"status": "scored", "candidate_coordinates": [[dataset, 0]], "source_result_sha256": source_result},
        SourceProvenGroupEvidence(evidence, sha256(checkpoints.canonical_json_bytes(evidence)).hexdigest()),
        *(sha256(f"{dataset}:{part}".encode()).hexdigest() for part in ("root", "progress", "record")),
        source_result,
        sha256(f"{dataset}:attestation".encode()).hexdigest(),
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


def _global_parity_context(config: Exp417FrozenConfig) -> FrozenRunContext:
    datasets = ("Alpha_normal_1", "Beta_normal_1")
    lanes = FrozenLaneSnapshot({}, {}, {}, {}, config=config, population=datasets)
    return FrozenRunContext(
        database_sha256=config.database_sha256,
        datasets=datasets,
        lanes=lanes,
        test_identity_manifest={dataset: tuple((index, 100) for index in range(4)) for dataset in datasets},
        code_commit="c" * 40,
        runtime_versions={"python": "test"},
        config=config,
    )


def _global_parity_result(
    context: FrozenRunContext, config: Exp417FrozenConfig, dataset: str, candidates: set[int],
) -> DatasetResult:
    group = GroupQualification(
        100, 100, "qualified", 0.0, 0.0, 1.0,
        {name: 1.0 for name in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")},
        True, True, False,
    )
    audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
    contract = FrozenTrainContract(dataset, (group,), audit, "a" * 64)

    def seed_decision(seed: int, candidate: bool) -> SeedDecision:
        score = HorizonScore(1, 1.0, 0.001 if candidate else 0.5, 0.5 if candidate else 0.0)
        scores = {horizon: HorizonScore(horizon, score.score, score.p_value, score.persistence) for horizon in (1, 2, 10)}
        return SeedDecision(seed, scores, candidate, config=config)

    decisions = tuple(
        TestInstanceDecision(
            dataset, index, 100,
            tuple(seed_decision(seed, index in candidates) for seed in config.model_seeds),
        )
        for index, _ in context.test_identity_manifest[dataset]
    )
    evidence = DatasetDecisionEvidence(
        dataset, context, contract, context.database_sha256, contract.train_identity_sha256, decisions,
    )
    shadows = tuple(
        ShadowCandidate(dataset, decision.instance_index, decision.length, decision.seed_decisions)
        for decision in decisions if decision.candidate
    )
    return DatasetResult(
        dataset, context.test_sizes[dataset], 1, (group,), shadows, evidence,
        context.lanes, context, "qualified",
    )


def _compact_import_record(result: DatasetResult) -> SimpleNamespace:
    coordinates = [[candidate.dataset, candidate.instance_index] for candidate in result.candidates]
    source = sha256(checkpoints.canonical_json_bytes({"status": "scored", "candidate_coordinates": coordinates})).hexdigest()
    return SimpleNamespace(
        dataset=result.dataset,
        origin="exp424_import",
        source_result_sha256=source,
        result_payload={"status": "scored", "candidate_coordinates": coordinates, "source_result_sha256": source},
    )


def _compact_store(context: FrozenRunContext, records: list[object]) -> SimpleNamespace:
    by_dataset = {getattr(record, "dataset"): record for record in records}
    return SimpleNamespace(
        context=context,
        expected_datasets=context.datasets,
        iter_records=lambda: iter(records),
        load_record=lambda dataset: by_dataset[dataset],
    )


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

    def test_heartbeat_write_failure_is_reraised_when_closed(self) -> None:
        """A background fsync failure must abort the foreground dataset before commit."""
        progress = SimpleNamespace(current_phase="preprocess", stage="running")
        writer = SimpleNamespace(write=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("fsync failed")))
        pump = _HeartbeatPump(writer, progress, interval=0.01)
        pump.start(); time.sleep(0.02)
        with self.assertRaisesRegex(OSError, "fsync failed"):
            pump.close()
        self.assertFalse(pump._thread.is_alive())

    def test_heartbeat_failure_publishes_failed_progress_without_native_commit(self) -> None:
        """A failed heartbeat cannot leave a native record or terminal completion behind."""
        context = _runtime_context(("Eligible_normal_1",))
        provenance = _runtime_provenance(context, "heartbeat-failure")
        dataset = context.datasets[0]
        train = RawTrainDataset(dataset, (RawSeries(10, tuple(float(index) for index in range(8))),))
        audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output, checkpoint_root = root / "output", root / "checkpoints"
            writer = AtomicProgressWriter.create(
                output_root=output, run_identity=provenance.run_identity, command="new",
                checkpoint_root=checkpoint_root, total_datasets=1, backend="mps",
            )
            writer.heartbeat_interval_seconds = 0.01
            real_write = writer.write

            def fail_heartbeat(progress: RuntimeProgress, *, event: str, detail: object = None) -> None:
                if event == "heartbeat":
                    raise OSError("heartbeat fsync failed")
                real_write(progress, event=event, detail=detail)

            writer.write = fail_heartbeat  # type: ignore[method-assign]
            store = Exp425CheckpointStore.create(checkpoint_root, provenance, context)
            record = _runtime_native_record(
                store, provenance, dataset, status="scored", coordinates=[[dataset, 1]], reason="native_mps",
            )
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            lock = os.open(root / "mps.lock", os.O_RDWR | os.O_CREAT, 0o600)

            def compute(*_args: object) -> tuple[NativeRecordEnvelope, object]:
                time.sleep(0.03)
                return record, object()

            with patch.object(Exp425DatabaseSnapshot, "verify_boundary"), patch.object(
                Exp425DatabaseSnapshot, "load_train", return_value=train
            ), patch.object(Exp425DatabaseSnapshot, "close"), patch(
                "experiments.exp425_selective_recovery.runtime._require_mps"
            ), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ), patch(
                "experiments.exp425_selective_recovery.runtime._compute_native", side_effect=compute
            ):
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
                result = run_selective_population(context, store, session)

            self.assertEqual(result.state, "failed")
            self.assertEqual(result.counts, Exp425RunCounts(1))
            self.assertEqual(tuple(store.iter_records()), ())
            events = [json.loads(line) for line in writer.paths.events.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events if event["event"] == "dataset_terminal"], [])
            self.assertEqual(events[-1]["event"], "failure")
            self.assertEqual(events[-1]["state"], "failed")

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
        audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
        store = SimpleNamespace(provenance=SimpleNamespace(run_identity="runtime-test"), run_context_sha256="d" * 64)
        runtime = SimpleNamespace(
            audit=audit,
            adoption=_mps_assessment(),
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
            audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
            lock = os.open(root / "lock", os.O_RDWR | os.O_CREAT, 0o600)
            with patch("experiments.exp425_selective_recovery.runtime._require_mps"), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ):
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
            with self.assertRaises(TypeError):
                copy.copy(session)
            with self.assertRaises(TypeError):
                pickle.dumps(session)
            with patch.object(Exp425DatabaseSnapshot, "close"):
                session.close()
            with self.assertRaisesRegex(ValueError, "closed"):
                session._require_live()
            with self.assertRaisesRegex(ValueError, "MPS audit"):
                build_exp425_runtime(snapshot, BackendAudit("mps", True, 0.0, 1.0, 1.1, False), _mps_assessment(), writer, "c" * 40)

    def test_runtime_session_requires_an_explicit_executed_head(self) -> None:
        for callable_ in (build_exp425_runtime, Exp425RuntimeSession):
            parameter = inspect.signature(callable_).parameters["executed_head"]
            self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_execute_rejects_provenance_context_commit_mismatch_before_new_roots(self) -> None:
        context = _runtime_context()
        provenance = replace(_runtime_provenance(context), git_commit="d" * 40)
        audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            with patch.object(Exp425DatabaseSnapshot, "close"):
                with self.assertRaisesRegex(ValueError, "git commit"):
                    execute_runtime_command(
                        "new", output_root=root / "output", checkpoint_root=root / "checkpoint",
                        provenance=provenance, context=context, snapshot=snapshot,
                        audit=audit, adoption=_mps_assessment(),
                    )
            self.assertFalse((root / "output").exists())
            self.assertFalse((root / "checkpoint").exists())

    def test_runtime_session_rejects_public_and_private_binding_reassignment(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root, run_identity="runtime-test", command="new",
                checkpoint_root=root / "checkpoints", total_datasets=1, backend="mps",
            )
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
            lock = os.open(root / "lock", os.O_RDWR | os.O_CREAT, 0o600)
            with patch("experiments.exp425_selective_recovery.runtime._require_mps"), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ):
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
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
            audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
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
            ) as native_mint, patch.object(exp417_qualification, "FROZEN_CONFIG", context.config):
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
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
                    "mps_adoption_sha256": record.mps_adoption.sha256,
                    "mps_execution_sha256": record.mps_execution.sha256,
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
            audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
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
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
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
                audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
                with patch.object(Exp425DatabaseSnapshot, "close"), patch(
                    "experiments.exp425_selective_recovery.runtime._require_mps"
                ), patch(
                    "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=first_lock
                ):
                    first_session = build_exp425_runtime(first_snapshot, audit, _mps_assessment(), writer, "c" * 40)
                    claim = {
                        "kind": "native", "run_identity": provenance.run_identity,
                        "dataset": dataset, "context_sha256": store.run_context_sha256,
                        "trace_sha256": record.trace_sha256,
                        "record_sha256": record.record_payload_sha256,
                        "backend_sha256": record.backend_audit_sha256,
                        "mps_adoption_sha256": record.mps_adoption.sha256,
                        "mps_execution_sha256": record.mps_execution.sha256,
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
                        ("mps_adoption_sha256", "4" * 64), ("mps_execution_sha256", "5" * 64),
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
                ), patch.object(exp417_qualification, "FROZEN_CONFIG", context.config):
                    resumed_session = build_exp425_runtime(second_snapshot, audit, _mps_assessment(), resumed_writer, "c" * 40)
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


class CompactGlobalWorkloadParityTests(unittest.TestCase):
    def _config(self, *, rate: float = 0.5, expected_hard: int = 0) -> Exp417FrozenConfig:
        return replace(
            FROZEN_CONFIG,
            expected_datasets=2,
            expected_hard=expected_hard,
            expected_standard=0,
            expected_priority=0,
            expected_automatic_normal=0,
            database_sha256="a" * 64,
            dataset_workload_rate=1.0,
            global_workload_rate=rate,
        )

    def test_compact_global_gate_matches_exp417_below_exact_and_above_boundaries(self) -> None:
        """Changing compact candidate accounting, comparison, or status loses Exp417 parity."""
        config = self._config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(
            exp417_contracts, "FROZEN_CONFIG", config
        ), patch.object(exp417_qualification, "FROZEN_CONFIG", config):
            context = _global_parity_context(config)
            cases = {
                "below": ({0, 1}, {0}),
                "exact": ({0, 1}, {0, 1}),
                "above": ({0, 1, 2}, {0, 1}),
            }
            for name, (alpha, beta) in cases.items():
                with self.subTest(name=name):
                    results = (
                        _global_parity_result(context, config, "Alpha_normal_1", alpha),
                        _global_parity_result(context, config, "Beta_normal_1", beta),
                    )
                    expected = apply_global_workload(results, 8, context=context)
                    store = _compact_store(context, [_compact_import_record(result) for result in results])
                    self.assertEqual(_apply_compact_global_workload(context, store), expected.status)

    def test_compact_global_gate_rejects_terminal_failed_result(self) -> None:
        config = self._config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(
            exp417_contracts, "FROZEN_CONFIG", config
        ), patch.object(exp417_qualification, "FROZEN_CONFIG", config):
            context = _global_parity_context(config)
            results = tuple(_global_parity_result(context, config, dataset, set()) for dataset in context.datasets)
            records = [_compact_import_record(result) for result in results]
            records[0].result_payload["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "failed"):
                _apply_compact_global_workload(context, _compact_store(context, records))

    def test_global_gate_failure_blocks_review_only_runtime_publication(self) -> None:
        """A compact global-failure status must become the existing terminal runtime failure."""
        context = _runtime_context(("Alpha_normal_1", "Beta_normal_1"))
        provenance = _runtime_provenance(context, "exp425-global-gate-failure")
        context_decoder = patch.object(checkpoints, "_context_from_payload", return_value=context)
        context_decoder.start()
        self.addCleanup(context_decoder.stop)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicProgressWriter.create(
                output_root=root / "output", run_identity=provenance.run_identity, command="new",
                checkpoint_root=root / "checkpoints", total_datasets=2, backend="mps",
            )
            store = Exp425CheckpointStore.create(root / "checkpoints", provenance, context)
            records = tuple(
                _runtime_native_record(store, provenance, dataset, status="scored",
                                       coordinates=[[dataset, 0], [dataset, 1]], reason="native_mps")
                for dataset in context.datasets
            )
            snapshot = object.__new__(Exp425DatabaseSnapshot)
            lock = os.open(root / "mps.lock", os.O_RDWR | os.O_CREAT, 0o600)
            audit = BackendAudit("cpu", True, 4.0948700075205124e-07, 0.9999999999845686, 0.6709759390840965, False)
            with patch("experiments.exp425_selective_recovery.runtime._require_mps"), patch(
                "experiments.exp425_selective_recovery.runtime._acquire_heavy_process_guard", return_value=lock
            ), patch.object(Exp425DatabaseSnapshot, "verify_boundary"), patch.object(
                Exp425DatabaseSnapshot, "close"
            ), patch.object(exp417_qualification, "FROZEN_CONFIG", context.config):
                session = build_exp425_runtime(snapshot, audit, _mps_assessment(), writer, "c" * 40)
                for record in records:
                    claim = capabilities._mint(
                        session, kind="native", run_identity=provenance.run_identity,
                        dataset=record.dataset, context_sha256=store.run_context_sha256,
                        trace_sha256=record.trace_sha256, record_sha256=record.record_payload_sha256,
                        backend_sha256=record.backend_audit_sha256,
                        mps_adoption_sha256=record.mps_adoption.sha256,
                        mps_execution_sha256=record.mps_execution.sha256,
                        process_nonce=session.process_nonce,
                    )
                    store._commit_native(claim, record)
                self.assertEqual(_apply_compact_global_workload(context, store), "global_workload_gate_failed")
                result = run_selective_population(context, store, session)
            self.assertEqual(result.state, "failed")
            self.assertEqual(result.global_workload_status, "failed")
            self.assertEqual(json.loads(writer.paths.progress.read_text())["state"], "failed")
            writer.close()

    def test_compact_global_gate_rejects_invalid_record_coverage_coordinates_and_schema(self) -> None:
        config = self._config()
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(
            exp417_contracts, "FROZEN_CONFIG", config
        ), patch.object(exp417_qualification, "FROZEN_CONFIG", config):
            context = _global_parity_context(config)
            results = tuple(_global_parity_result(context, config, dataset, {0, 1}) for dataset in context.datasets)

            def records() -> list[SimpleNamespace]:
                return [_compact_import_record(result) for result in results]

            mutations = {
                "duplicate": lambda value: value[0].result_payload.__setitem__("candidate_coordinates", [["Alpha_normal_1", 0], ["Alpha_normal_1", 0]]),
                "unsorted": lambda value: value[0].result_payload.__setitem__("candidate_coordinates", [["Alpha_normal_1", 1], ["Alpha_normal_1", 0]]),
                "wrong dataset": lambda value: value[0].result_payload.__setitem__("candidate_coordinates", [["Beta_normal_1", 0]]),
                "boolean index": lambda value: value[0].result_payload.__setitem__("candidate_coordinates", [["Alpha_normal_1", True]]),
                "outside manifest": lambda value: value[0].result_payload.__setitem__("candidate_coordinates", [["Alpha_normal_1", 9]]),
                "candidate on abstained": lambda value: value[0].result_payload.update(status="abstained"),
                "unknown status": lambda value: value[0].result_payload.update(status="unknown"),
                "unknown key": lambda value: value[0].result_payload.__setitem__("unexpected", True),
                "missing source": lambda value: value[0].result_payload.pop("source_result_sha256"),
                "wrong source": lambda value: value[0].result_payload.update(source_result_sha256="0" * 64),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    value = records()
                    mutate(value)
                    with self.assertRaises(ValueError):
                        _apply_compact_global_workload(context, _compact_store(context, value))

            with self.subTest(name="incomplete coverage"):
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(context, _compact_store(context, records()[:1]))
            with self.subTest(name="wrong sorted coverage"):
                value = records()
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(context, _compact_store(context, list(reversed(value))))
            with self.subTest(name="store total mismatch"):
                store = _compact_store(context, records())
                store.expected_datasets = ("Alpha_normal_1",)
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(context, store)
            with self.subTest(name="frozen TEST total mismatch"):
                mismatched_total = replace(context)
                object.__setattr__(mismatched_total, "_test_sizes", MappingProxyType({
                    "Alpha_normal_1": 3, "Beta_normal_1": 4,
                }))
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(mismatched_total, _compact_store(mismatched_total, records()))

            source = sha256(checkpoints.canonical_json_bytes({"status": "scored", "candidate_coordinates": []})).hexdigest()
            native = SimpleNamespace(
                dataset="Alpha_normal_1", origin="exp425_native", result_evidence_sha256="c" * 64,
                result_payload={"effective_result": {"status": "scored", "candidate_coordinates": [],
                                                        "result_evidence_sha256": "c" * 64,
                                                        "result_source_sha256": source},
                                "native_trace_body": {}},
            )
            clean = records()[1]
            with self.subTest(name="native missing source"):
                native.result_payload["effective_result"].pop("result_source_sha256")
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(context, _compact_store(context, [native, clean]))
            with self.subTest(name="native unknown key"):
                native.result_payload["effective_result"]["result_source_sha256"] = source
                native.result_payload["effective_result"]["unexpected"] = True
                with self.assertRaises(ValueError):
                    _apply_compact_global_workload(context, _compact_store(context, [native, clean]))

    def test_compact_global_gate_rejects_lane_overlap_and_context_cap_mismatch(self) -> None:
        config = self._config(expected_hard=1)
        with patch.object(Exp417FrozenConfig, "validate", lambda _self: None), patch.object(
            exp417_contracts, "FROZEN_CONFIG", config
        ), patch.object(exp417_qualification, "FROZEN_CONFIG", config):
            context = FrozenRunContext(
                database_sha256=config.database_sha256,
                datasets=("Alpha_normal_1", "Beta_normal_1"),
                lanes=FrozenLaneSnapshot({"Alpha_normal_1": (0,)}, {}, {}, {}, config=config, population=("Alpha_normal_1", "Beta_normal_1")),
                test_identity_manifest={dataset: tuple((index, 100) for index in range(4)) for dataset in ("Alpha_normal_1", "Beta_normal_1")},
                code_commit="c" * 40,
                runtime_versions={"python": "test"},
                config=config,
            )
            records = [
                SimpleNamespace(dataset="Alpha_normal_1", origin="exp424_import", source_result_sha256="a" * 64,
                                result_payload={"status": "scored", "candidate_coordinates": [["Alpha_normal_1", 0]], "source_result_sha256": "a" * 64}),
                SimpleNamespace(dataset="Beta_normal_1", origin="exp424_import", source_result_sha256="b" * 64,
                                result_payload={"status": "abstained", "candidate_coordinates": [], "source_result_sha256": "b" * 64}),
            ]
            with self.assertRaises(ValueError):
                _apply_compact_global_workload(context, _compact_store(context, records))
            mismatched = replace(context, config=replace(config, global_workload_rate=0.25))
            clean_records = [
                SimpleNamespace(dataset=dataset, origin="exp424_import", source_result_sha256=digest,
                                result_payload={"status": "abstained", "candidate_coordinates": [], "source_result_sha256": digest})
                for dataset, digest in (("Alpha_normal_1", "a" * 64), ("Beta_normal_1", "b" * 64))
            ]
            with self.assertRaisesRegex(ValueError, "cap"):
                _apply_compact_global_workload(mismatched, _compact_store(mismatched, clean_records))
