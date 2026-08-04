"""Compact shared-context checkpoint tests for Exp425."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
from collections.abc import Mapping

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.checkpoints import _context_payload, _context_from_payload
from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import _context
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery import checkpoints
from experiments.exp425_selective_recovery.contracts import (
    Exp425RunProvenance,
    ImportedRecordEnvelope,
    NativeRecordEnvelope,
    SourceProvenGroupEvidence,
)
from experiments.exp425_selective_recovery import capabilities


def _digest(letter: str) -> str:
    return letter * 64


def _checkpoint_snapshot(root: Path) -> dict[str, tuple[bytes | None, tuple[int, int, int, int, int, int, int]]]:
    """All application-observed metadata except atime, plus bytes for regular files."""
    snapshot: dict[str, tuple[bytes | None, tuple[int, int, int, int, int, int, int]]] = {}
    for path in (root, *sorted(root.rglob("*"))):
        info = path.lstat()
        metadata = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_nlink, info.st_mtime_ns, info.st_ctime_ns)
        snapshot[str(path.relative_to(root)) if path != root else "."] = (path.read_bytes() if stat.S_ISREG(info.st_mode) else None, metadata)
    return snapshot


def _provenance(context: FrozenRunContext) -> Exp425RunProvenance:
    return Exp425RunProvenance(
        run_identity="exp425-test-run", git_commit="c" * 40,
        database_sha256=context.database_sha256,
        lane_artifact_sha256={"hard": _digest("a"), "standard": _digest("b"), "priority": _digest("c"), "automatic_normal": _digest("d")},
        frozen_config_sha256=_digest("e"), fold_contract_sha256=_digest("f"),
        model_contract_sha256=_digest("1"), backend_contract_sha256=_digest("2"),
        model_seeds=(1,), perturbation_seeds=(2,), runtime_versions={"python": "test"},
        expected_datasets=context.datasets, predecessor_compatibility_sha256=_digest("3"),
    )


def _native_record_for(store: Exp425CheckpointStore, provenance: Exp425RunProvenance, dataset: str) -> NativeRecordEnvelope:
    values = {name: sha256(f"{dataset}:{name}".encode()).hexdigest() for name in (
        "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256",
        "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256",
        "result_evidence_sha256", "backend_audit_sha256",
    )}
    trace_body = {
        "schema": "exp425.native-trace.v1", "run_identity": provenance.run_identity,
        "dataset": dataset, "run_context_sha256": store.run_context_sha256,
        **values, "device": "mps",
    }
    trace = sha256(canonical_json_bytes(trace_body)).hexdigest()
    result_payload = {
        "effective_result": {"status": "abstained", "result_evidence_sha256": values["result_evidence_sha256"]},
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
        "schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native",
        "run_context_sha256": store.run_context_sha256, "result_payload": result_payload,
        "group_evidence": group_evidence, "trace_sha256": trace, "model_hashes": model_hashes,
        "calibration_hashes": calibration_hashes, "backend_audit_sha256": values["backend_audit_sha256"],
        "elapsed_seconds": 0.0, "reason": "synthetic",
    }
    return NativeRecordEnvelope(
        dataset, result_payload, group_evidence, trace_sha256=trace,
        record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(),
        model_hashes=model_hashes, calibration_hashes=calibration_hashes,
        elapsed_seconds=0.0, reason="synthetic", **values,
    )


def _native_capability(store: Exp425CheckpointStore, provenance: Exp425RunProvenance, authority: object, record: NativeRecordEnvelope) -> object:
    return capabilities._mint(
        authority, kind="native", run_identity=provenance.run_identity, dataset=record.dataset,
        context_sha256=store.run_context_sha256, trace_sha256=record.trace_sha256,
        record_sha256=record.record_payload_sha256, backend_sha256=record.backend_audit_sha256,
    )


def _import_record_for(dataset: str, index: int, attestation: str) -> ImportedRecordEnvelope:
    evidence = {"dataset": dataset, "source_groups": [index]}
    return ImportedRecordEnvelope(
        dataset, {"status": "abstained", "source_index": index},
        SourceProvenGroupEvidence(evidence, sha256(canonical_json_bytes(evidence)).hexdigest()),
        sha256(f"root:{dataset}".encode()).hexdigest(), sha256(f"progress:{dataset}".encode()).hexdigest(),
        sha256(f"record:{dataset}".encode()).hexdigest(), sha256(f"result:{dataset}".encode()).hexdigest(),
        attestation, float(index), "sealed local exp424 import",
    )


def _import_stage_digest(store: Exp425CheckpointStore, record: ImportedRecordEnvelope) -> str:
    return sha256(canonical_json_bytes(checkpoints._import_record_body(store.run_context_sha256, record))).hexdigest()


def _import_final_attestation(store: Exp425CheckpointStore, batch_identity: str, attestation: str, records: tuple[ImportedRecordEnvelope, ...]) -> str:
    return sha256(canonical_json_bytes({
        "batch_identity": batch_identity,
        "record_digests": dict(sorted((record.dataset, _import_stage_digest(store, record)) for record in records)),
        "attestation_sha256": attestation,
    })).hexdigest()


def _begin_and_stage_imports(store: Exp425CheckpointStore, provenance: Exp425RunProvenance, authority: object, nonce: str, batch_identity: str, attestation: str, records: tuple[ImportedRecordEnvelope, ...]) -> object:
    batch = store._begin_import_batch(capabilities._mint(
        authority, kind="import_begin", run_identity=provenance.run_identity, batch_identity=batch_identity,
        attestation_sha256=attestation, process_nonce=nonce,
    ), batch_identity)
    for record in records:
        store._stage_imported(capabilities._mint(
            authority, kind="import_stage", run_identity=provenance.run_identity, batch_identity=batch_identity,
            dataset=record.dataset, source_sha256=record.source_record_sha256,
            attestation_sha256=attestation, process_nonce=nonce, record_sha256=_import_stage_digest(store, record),
        ), batch, record)
    return batch


def _promote_imports(store: Exp425CheckpointStore, provenance: Exp425RunProvenance, authority: object, nonce: str, batch_identity: str, attestation: str, records: tuple[ImportedRecordEnvelope, ...], batch: object) -> None:
    store._promote_import_batch(capabilities._mint(
        authority, kind="import_promote", run_identity=provenance.run_identity, batch_identity=batch_identity,
        attestation_sha256=attestation, process_nonce=nonce,
        final_attestation_sha256=_import_final_attestation(store, batch_identity, attestation, records),
    ), batch)


@dataclass(frozen=True)
class _CheckpointBenchmarkMetrics:
    repeated_context_bytes: int
    exp425_bytes: int
    byte_reduction_fraction: float
    old_median_validated_cycle_seconds: float
    exp425_median_open_iter_validated_cycle_seconds: float
    speedup: float
    old_candidate_coordinates: tuple[tuple[object, ...], ...]
    exp425_candidate_coordinates: tuple[tuple[object, ...], ...]
    old_result_evidence_digests: tuple[str, ...]
    exp425_result_evidence_digests: tuple[str, ...]
    completed_record_count: int


def _measure_shared_context_checkpoint_benchmark() -> _CheckpointBenchmarkMetrics:
    """Measure two fully validated 25-record checkpoint representations."""
    context = _context()
    provenance = _provenance(context)
    datasets = context.datasets[:25]

    def record_for(dataset: str, index: int, store: Exp425CheckpointStore) -> NativeRecordEnvelope:
        digest = lambda label: sha256(f"{dataset}:{label}".encode()).hexdigest()
        values = {name: digest(name) for name in (
            "train_identity_sha256", "eligibility_sha256", "processed_groups_sha256",
            "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256",
            "result_evidence_sha256", "backend_audit_sha256",
        )}
        trace_body = {
            "schema": "exp425.native-trace.v1", "run_identity": provenance.run_identity,
            "dataset": dataset, "run_context_sha256": store.run_context_sha256,
            **values, "device": "mps",
        }
        trace = sha256(canonical_json_bytes(trace_body)).hexdigest()
        result = {
            "effective_result": {
                "candidate_coordinates": [[dataset, index]],
                "result_evidence_sha256": values["result_evidence_sha256"],
                "status": "abstained",
            },
            "native_trace_body": trace_body,
        }
        body = {
            "schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native",
            "run_context_sha256": store.run_context_sha256, "result_payload": result,
            "group_evidence": {"eligibility_sha256": values["eligibility_sha256"], "processed_groups_sha256": values["processed_groups_sha256"]},
            "trace_sha256": trace, "model_hashes": {"train_identity_sha256": values["train_identity_sha256"], "model_evidence_sha256": values["model_evidence_sha256"], "frozen_train_contract_sha256": values["frozen_train_contract_sha256"]},
            "calibration_hashes": {"scoring_evidence_sha256": values["scoring_evidence_sha256"]},
            "backend_audit_sha256": values["backend_audit_sha256"], "elapsed_seconds": float(index),
            "reason": "benchmark",
        }
        return NativeRecordEnvelope(
            dataset, result, {"eligibility_sha256": values["eligibility_sha256"], "processed_groups_sha256": values["processed_groups_sha256"]},
            trace_sha256=trace, record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(),
            model_hashes={"train_identity_sha256": values["train_identity_sha256"], "model_evidence_sha256": values["model_evidence_sha256"], "frozen_train_contract_sha256": values["frozen_train_contract_sha256"]},
            calibration_hashes={"scoring_evidence_sha256": values["scoring_evidence_sha256"]},
            elapsed_seconds=float(index), reason="benchmark", **values,
        )

    def checked_native_result(record: NativeRecordEnvelope) -> tuple[tuple[tuple[object, ...], ...], str]:
        payload = dict(record.result_payload)
        trace_body = payload.get("native_trace_body")
        effective = payload.get("effective_result")
        if not isinstance(trace_body, Mapping) or not isinstance(effective, Mapping):
            raise AssertionError("benchmark native result payload is invalid")
        if sha256(canonical_json_bytes(checkpoints.thaw_payload(trace_body))).hexdigest() != record.trace_sha256:
            raise AssertionError("benchmark trace does not match its stored declared digest")
        result_evidence = effective.get("result_evidence_sha256")
        coordinates = effective.get("candidate_coordinates")
        if (
            not isinstance(result_evidence, str)
            or trace_body.get("result_evidence_sha256") != result_evidence
            or not isinstance(coordinates, tuple)
            or any(not isinstance(row, tuple) for row in coordinates)
        ):
            raise AssertionError("benchmark effective result evidence is invalid")
        return tuple(tuple(row) for row in coordinates), result_evidence

    with TemporaryDirectory() as directory:
        root = Path(directory) / "checkpoints"
        store = Exp425CheckpointStore.create(root, provenance, context)
        authority = object()
        capabilities._register_authority(authority, "exp425_native")
        try:
            records = [record_for(dataset, index, store) for index, dataset in enumerate(datasets)]
            for record in records:
                token = capabilities._mint(
                    authority, kind="native", run_identity=provenance.run_identity,
                    dataset=record.dataset, context_sha256=store.run_context_sha256,
                    trace_sha256=record.trace_sha256, record_sha256=record.record_payload_sha256,
                    backend_sha256=record.backend_audit_sha256,
                )
                store._commit_native(token, record)
        finally:
            capabilities._close_authority(authority)
        repeated_records = [canonical_json_bytes({
            "schema": "exp424.repeated-context.v1", "context": _context_payload(context),
            "record": checkpoints._native_record_body(store.run_context_sha256, record) | {
                "record_payload_sha256": record.record_payload_sha256,
            },
        }) for record in records]
        repeated_context_bytes = sum(map(len, repeated_records))
        exp425_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

        def old_cycle() -> tuple[tuple[tuple[object, ...], ...], tuple[str, ...]]:
            coordinates: list[tuple[object, ...]] = []
            result_evidence_digests: list[str] = []
            for raw in repeated_records:
                payload = checkpoints._canonical_object(raw, "repeated benchmark record")
                if set(payload) != {"schema", "context", "record"} or payload["schema"] != "exp424.repeated-context.v1":
                    raise AssertionError("repeated benchmark schema is invalid")
                if _context_from_payload(payload["context"]) != context:
                    raise AssertionError("repeated benchmark context is invalid")
                persisted = payload["record"]
                if not isinstance(persisted, dict):
                    raise AssertionError("repeated benchmark record is invalid")
                decoded = checkpoints._decode_record(canonical_json_bytes(persisted), context)
                if not isinstance(decoded, NativeRecordEnvelope):
                    raise AssertionError("repeated benchmark record origin is invalid")
                rows, evidence = checked_native_result(decoded)
                coordinates.extend(rows)
                result_evidence_digests.append(evidence)
            return tuple(coordinates), tuple(result_evidence_digests)

        def new_cycle() -> tuple[tuple[tuple[object, ...], ...], tuple[str, ...], int]:
            reopened = Exp425CheckpointStore.open(root, provenance, context)
            coordinates: list[tuple[object, ...]] = []
            result_evidence_digests: list[str] = []
            for record in reopened.iter_records():
                if not isinstance(record, NativeRecordEnvelope):
                    raise AssertionError("Exp425 benchmark record origin is invalid")
                rows, evidence = checked_native_result(record)
                coordinates.extend(rows)
                result_evidence_digests.append(evidence)
            return tuple(coordinates), tuple(result_evidence_digests), len(reopened.completed_record_digests())

        old_coordinates, old_evidence = old_cycle()
        exp425_coordinates, exp425_evidence, completed_count = new_cycle()
        old_samples: list[float] = []
        exp425_samples: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            old_coordinates, old_evidence = old_cycle()
            old_samples.append(time.perf_counter() - started)
            started = time.perf_counter()
            exp425_coordinates, exp425_evidence, completed_count = new_cycle()
            exp425_samples.append(time.perf_counter() - started)
        old_seconds = sorted(old_samples)[2]
        exp425_seconds = sorted(exp425_samples)[2]
        return _CheckpointBenchmarkMetrics(
            repeated_context_bytes=repeated_context_bytes,
            exp425_bytes=exp425_bytes,
            byte_reduction_fraction=1.0 - (exp425_bytes / repeated_context_bytes),
            old_median_validated_cycle_seconds=old_seconds,
            exp425_median_open_iter_validated_cycle_seconds=exp425_seconds,
            speedup=old_seconds / exp425_seconds,
            old_candidate_coordinates=old_coordinates,
            exp425_candidate_coordinates=exp425_coordinates,
            old_result_evidence_digests=old_evidence,
            exp425_result_evidence_digests=exp425_evidence,
            completed_record_count=completed_count,
        )


class CheckpointStoreTests(unittest.TestCase):
    """Changing a record or shared context invalidates the persisted checkpoint."""

    def test_create_persists_one_shared_context_and_read_only_inspection(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            self.assertEqual(
                {path.name for path in root.iterdir()},
                {"checkpoint_provenance.json", "run_context.json", "progress.json", "datasets", "transactions", "import_batches"},
            )
            raw = (root / "run_context.json").read_bytes()
            self.assertEqual(store.run_context_sha256, sha256(raw).hexdigest())
            before = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
            inspection = Exp425CheckpointStore.inspect_read_only(root, expected_run_identity="exp425-test-run")
            after = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
            self.assertEqual(inspection.run_identity, "exp425-test-run")
            self.assertEqual(inspection.completed_count, 0)
            self.assertEqual(before, after)

    def test_inspect_read_only_preserves_all_observed_bytes_and_metadata_on_success_and_failure(self) -> None:
        """Any inspect write, reconciliation, or quarantine would change this complete snapshot."""
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, provenance, context)
            before = _checkpoint_snapshot(root)
            Exp425CheckpointStore.inspect_read_only(root)
            self.assertEqual(_checkpoint_snapshot(root), before)
            (root / "unexpected-entry").write_bytes(b"foreign")
            before_failure = _checkpoint_snapshot(root)
            with self.assertRaises(ValueError):
                Exp425CheckpointStore.inspect_read_only(root)
            self.assertEqual(_checkpoint_snapshot(root), before_failure)

    def test_new_and_open_enforce_root_lifecycle(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            with self.assertRaises(ValueError):
                Exp425CheckpointStore.open(root, provenance, context)
            Exp425CheckpointStore.create(root, provenance, context)
            with self.assertRaises(ValueError):
                Exp425CheckpointStore.create(root, provenance, context)
            reopened = Exp425CheckpointStore.open(root, provenance, context)
            self.assertEqual(reopened.completed_record_digests(), {})

    def test_inspection_rejects_noncanonical_progress_without_repairing_it(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            progress = root / "progress.json"
            corrupt = b'{ "completed_count": 0}'
            progress.write_bytes(corrupt)
            before = progress.read_bytes()
            with self.assertRaises(ValueError):
                Exp425CheckpointStore.inspect_read_only(root)
            self.assertEqual(progress.read_bytes(), before)

    def test_progress_rejects_uppercase_digests_and_completed_current_dataset(self) -> None:
        """Removing strict progress semantics would admit canonically rehashed invalid resume state."""
        context = _context()
        provenance = _provenance(context)
        for name, mutate in (
            ("uppercase digest", lambda progress, dataset: progress["record_digests"].__setitem__(dataset, "A" * 64)),
            ("completed current", lambda progress, dataset: progress.__setitem__("current_dataset", dataset)),
        ):
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                store = Exp425CheckpointStore.create(root, provenance, context)
                record = _native_record_for(store, provenance, context.datasets[0])
                authority = object()
                capabilities._register_authority(authority, "exp425_native")
                try:
                    store._commit_native(_native_capability(store, provenance, authority, record), record)
                finally:
                    capabilities._close_authority(authority)
                path = root / "progress.json"
                progress = json.loads(path.read_bytes())
                mutate(progress, record.dataset)
                body = dict(progress)
                body.pop("payload_sha256")
                progress["payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
                path.write_bytes(canonical_json_bytes(progress))
                with self.assertRaises(ValueError):
                    Exp425CheckpointStore.inspect_read_only(root)

    def test_native_write_requires_live_exact_capability_and_injects_context_on_read(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            store = Exp425CheckpointStore.create(Path(directory) / "checkpoints", provenance, context)
            dataset = context.datasets[0]
            values = {name: _digest(letter) for name, letter in (
                ("train_identity_sha256", "4"), ("eligibility_sha256", "5"), ("processed_groups_sha256", "6"),
                ("model_evidence_sha256", "7"), ("frozen_train_contract_sha256", "8"), ("scoring_evidence_sha256", "9"),
                ("result_evidence_sha256", "a"), ("backend_audit_sha256", "b"),
            )}
            trace_body = {"schema": "exp425.native-trace.v1", "run_identity": provenance.run_identity, "dataset": dataset,
                "run_context_sha256": store.run_context_sha256, **values, "device": "mps"}
            trace = sha256(canonical_json_bytes(trace_body)).hexdigest()
            result_payload = {"effective_result": {"status": "abstained", "result_evidence_sha256": values["result_evidence_sha256"]}, "native_trace_body": trace_body}
            group_evidence = {"eligibility_sha256": values["eligibility_sha256"], "processed_groups_sha256": values["processed_groups_sha256"]}
            model_hashes = {"train_identity_sha256": values["train_identity_sha256"], "model_evidence_sha256": values["model_evidence_sha256"], "frozen_train_contract_sha256": values["frozen_train_contract_sha256"]}
            calibration_hashes = {"scoring_evidence_sha256": values["scoring_evidence_sha256"]}
            body = {"schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native",
                "run_context_sha256": store.run_context_sha256, "result_payload": result_payload, "group_evidence": group_evidence,
                "trace_sha256": trace, "model_hashes": model_hashes, "calibration_hashes": calibration_hashes, "backend_audit_sha256": values["backend_audit_sha256"],
                "elapsed_seconds": 0.0, "reason": "synthetic"}
            record = NativeRecordEnvelope(dataset, result_payload, group_evidence, trace_sha256=trace,
                record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(), model_hashes=model_hashes, calibration_hashes=calibration_hashes, elapsed_seconds=0.0,
                reason="synthetic", **values)
            with self.assertRaises(ValueError):
                store._commit_native(object(), record)
            authority = object(); capabilities._register_authority(authority, "local")
            try:
                token = capabilities._mint(authority, kind="native", run_identity=provenance.run_identity, dataset=dataset,
                    context_sha256=store.run_context_sha256, trace_sha256=trace, record_sha256=record.record_payload_sha256,
                    backend_sha256=record.backend_audit_sha256)
                self.assertEqual(store._commit_native(token, record), record.record_payload_sha256)
            finally:
                capabilities._close_authority(authority)
            loaded = store.load_record(dataset)
            self.assertIs(loaded.context, context)
            raw = (store.root / "datasets" / __import__("experiments.exp425_selective_recovery.checkpoints", fromlist=["record_filename"]).record_filename(dataset)).read_bytes()
            self.assertNotIn(b'"test_identity_manifest"', raw)
            self.assertLess(len(raw), len(canonical_json_bytes(_context_payload(context))) // 5)

    def test_native_recovery_advances_each_durable_phase_once(self) -> None:
        """Native B→E crash recovery removes or adopts only the durable state."""
        class SimulatedProcessDeath(RuntimeError):
            pass

        context = _context()
        provenance = _provenance(context)
        for phase in ("intent_durable", "record_durable", "progress_durable", "intent_removed"):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                store = Exp425CheckpointStore.create(root, provenance, context)
                record = _native_record_for(store, provenance, context.datasets[0])
                authority = object()
                capabilities._register_authority(authority, "exp425_native")
                try:
                    with patch.object(checkpoints, "_after_native_phase", side_effect=lambda reached: (
                        (_ for _ in ()).throw(SimulatedProcessDeath(reached)) if reached == phase else None
                    )):
                        with self.assertRaisesRegex(SimulatedProcessDeath, phase):
                            store._commit_native(_native_capability(store, provenance, authority, record), record)
                finally:
                    capabilities._close_authority(authority)
                reopened = Exp425CheckpointStore.open(root, provenance, context)
                expected = {} if phase == "intent_durable" else {record.dataset: record.record_payload_sha256}
                self.assertEqual(dict(reopened.completed_record_digests()), expected)
                self.assertEqual(list((root / "transactions").iterdir()), [])

    def test_native_intent_requires_exact_schema_origin_and_operation(self) -> None:
        context = _context()
        provenance = _provenance(context)
        dataset = context.datasets[0]
        base = {
            "schema": "exp425.intent.v1", "operation": "commit_native", "origin": "exp425_native",
            "dataset": dataset, "record_payload_sha256": _digest("a"),
        }
        malformed = (
            {key: value for key, value in base.items() if key != "operation"},
            base | {"operation": "other"},
            base | {"extra": "rejected"},
        )
        for intent in malformed:
            with self.subTest(intent=intent), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                Exp425CheckpointStore.create(root, provenance, context)
                (root / "transactions" / "native.json").write_bytes(canonical_json_bytes(intent))
                with self.assertRaises(ValueError):
                    Exp425CheckpointStore.open(root, provenance, context)

    def test_native_load_rejects_trace_literals_and_cross_field_mutations(self) -> None:
        context = _context()
        provenance = _provenance(context)
        mutations = (
            ("run identity", lambda payload: payload["result_payload"]["native_trace_body"].__setitem__("run_identity", "other-run")),
            ("dataset", lambda payload: payload["result_payload"]["native_trace_body"].__setitem__("dataset", "other-dataset")),
            ("context", lambda payload: payload["result_payload"]["native_trace_body"].__setitem__("run_context_sha256", _digest("0"))),
            ("device", lambda payload: payload["result_payload"]["native_trace_body"].__setitem__("device", "cpu")),
            ("backend", lambda payload: payload["result_payload"]["native_trace_body"].__setitem__("backend_audit_sha256", _digest("0"))),
            ("result", lambda payload: payload["result_payload"]["effective_result"].__setitem__("result_evidence_sha256", _digest("0"))),
            ("eligibility", lambda payload: payload["group_evidence"].__setitem__("eligibility_sha256", _digest("0"))),
            ("processed groups", lambda payload: payload["group_evidence"].__setitem__("processed_groups_sha256", _digest("0"))),
            ("train", lambda payload: payload["model_hashes"].__setitem__("train_identity_sha256", _digest("0"))),
            ("model", lambda payload: payload["model_hashes"].__setitem__("model_evidence_sha256", _digest("0"))),
            ("frozen train", lambda payload: payload["model_hashes"].__setitem__("frozen_train_contract_sha256", _digest("0"))),
            ("scoring", lambda payload: payload["calibration_hashes"].__setitem__("scoring_evidence_sha256", _digest("0"))),
            ("test label", lambda payload: payload["result_payload"]["effective_result"].__setitem__("TEST_label", "forbidden")),
            ("ground truth", lambda payload: payload["group_evidence"].__setitem__("ground_truth", "forbidden")),
            ("family test performance", lambda payload: payload["model_hashes"].__setitem__("family_TEST_performance", "forbidden")),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                store = Exp425CheckpointStore.create(root, provenance, context)
                record = _native_record_for(store, provenance, context.datasets[0])
                authority = object()
                capabilities._register_authority(authority, "exp425_native")
                try:
                    store._commit_native(_native_capability(store, provenance, authority, record), record)
                finally:
                    capabilities._close_authority(authority)
                path = root / "datasets" / checkpoints.record_filename(record.dataset)
                payload = json.loads(path.read_bytes())
                mutate(payload)
                trace = payload["result_payload"]["native_trace_body"]
                payload["trace_sha256"] = sha256(canonical_json_bytes(trace)).hexdigest()
                body = dict(payload)
                body.pop("record_payload_sha256")
                payload["record_payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
                path.write_bytes(canonical_json_bytes(payload))
                progress_path = root / "progress.json"
                progress = json.loads(progress_path.read_bytes())
                progress["record_digests"][record.dataset] = payload["record_payload_sha256"]
                progress_body = dict(progress)
                progress_body.pop("payload_sha256")
                progress["payload_sha256"] = sha256(canonical_json_bytes(progress_body)).hexdigest()
                progress_path.write_bytes(canonical_json_bytes(progress))
                with self.assertRaises(ValueError):
                    store.load_record(record.dataset)

    def test_lazy_open_defers_unknown_native_result_status_to_explicit_load(self) -> None:
        """Removing result-status validation would accept a canonically rehashed unknown outcome."""
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            record = _native_record_for(store, provenance, context.datasets[0])
            authority = object()
            capabilities._register_authority(authority, "exp425_native")
            try:
                store._commit_native(_native_capability(store, provenance, authority, record), record)
            finally:
                capabilities._close_authority(authority)
            path = root / "datasets" / checkpoints.record_filename(record.dataset)
            payload = json.loads(path.read_bytes())
            payload["result_payload"]["effective_result"]["status"] = "invented_status"
            body = dict(payload)
            body.pop("record_payload_sha256")
            payload["record_payload_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
            path.write_bytes(canonical_json_bytes(payload))
            progress_path = root / "progress.json"
            progress = json.loads(progress_path.read_bytes())
            progress["record_digests"][record.dataset] = payload["record_payload_sha256"]
            progress_body = dict(progress)
            progress_body.pop("payload_sha256")
            progress["payload_sha256"] = sha256(canonical_json_bytes(progress_body)).hexdigest()
            progress_path.write_bytes(canonical_json_bytes(progress))
            reopened = Exp425CheckpointStore.open(root, provenance, context)
            with self.assertRaisesRegex(ValueError, "result status"):
                reopened.load_record(record.dataset)

    def test_mutating_open_is_lazy_but_explicit_load_decodes_completed_record(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            record = _native_record_for(store, provenance, context.datasets[0])
            authority = object()
            capabilities._register_authority(authority, "exp425_native")
            try:
                store._commit_native(_native_capability(store, provenance, authority, record), record)
            finally:
                capabilities._close_authority(authority)
            with patch.object(checkpoints, "_decode_record", side_effect=AssertionError("open decoded completed record")):
                reopened = Exp425CheckpointStore.open(root, provenance, context)
            self.assertEqual(dict(reopened.completed_record_digests()), {record.dataset: record.record_payload_sha256})
            self.assertEqual(reopened.load_record(record.dataset).dataset, record.dataset)

    def test_store_lifetime_root_binding_rejects_replacement_before_public_operation(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            reopened = Exp425CheckpointStore.open(root, provenance, context)
            displaced = Path(directory) / "displaced"
            root.rename(displaced)
            root.mkdir()
            for instance in (store, reopened):
                with self.assertRaisesRegex(ValueError, "store root identity"):
                    instance.completed_record_digests()
            record = _native_record_for(store, provenance, context.datasets[0])
            authority = object()
            capabilities._register_authority(authority, "exp425_native")
            try:
                with self.assertRaisesRegex(ValueError, "store root identity"):
                    store._commit_native(_native_capability(store, provenance, authority, record), record)
            finally:
                capabilities._close_authority(authority)
            self.assertEqual(list(root.iterdir()), [])

    def test_quarantine_moves_only_the_exact_locked_root_inode(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            with checkpoints._locked_root(root, exclusive=True) as binding:
                quarantined = checkpoints._quarantine_owned(root, binding)
            self.assertFalse(root.exists())
            self.assertTrue(quarantined.is_dir())

    def test_quarantine_refuses_replaced_root_or_existing_destination(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            collision = Path(directory) / "collision"
            collision.mkdir()
            with checkpoints._locked_root(root, exclusive=True) as binding:
                with self.assertRaises(ValueError):
                    checkpoints._quarantine_owned(root, binding, destination=collision)
            self.assertTrue(root.is_dir())
            self.assertTrue(collision.is_dir())

    def test_atomic_quarantine_destination_race_preserves_source_and_racing_destination(self) -> None:
        """Replacing exclusive rename with overwrite rename would destroy the racer's destination."""
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            destination = Path(directory) / "race-destination"
            source_before = _checkpoint_snapshot(root)

            def race(parent_fd: int, source: str, target: str) -> None:
                destination.mkdir()
                (destination / "racer").write_bytes(b"racer-owned")
                raise ValueError("quarantine destination is unavailable")

            with checkpoints._locked_root(root, exclusive=True) as binding:
                with patch.object(checkpoints, "_rename_exclusive_at", side_effect=race):
                    with self.assertRaisesRegex(ValueError, "destination"):
                        checkpoints._quarantine_owned(root, binding, destination=destination)
            self.assertEqual(_checkpoint_snapshot(root), source_before)
            self.assertEqual((destination / "racer").read_bytes(), b"racer-owned")

    def test_open_quarantines_corruption_only_while_the_owned_root_is_bound(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            (root / "progress.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                Exp425CheckpointStore.open(root, _provenance(context), context)
            self.assertFalse(root.exists())
            self.assertEqual(len(list(Path(directory).glob("checkpoints.quarantine-*"))), 1)

    def test_root_and_child_directory_replacement_fail_before_mutation(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            with checkpoints._locked_root(root, exclusive=True) as binding:
                for name in ("datasets", "transactions"):
                    descriptor, identity = checkpoints._open_directory_at(binding, name)
                    original = root / name
                    displaced = root / (name + ".old")
                    original.rename(displaced)
                    original.mkdir()
                    try:
                        with self.assertRaises(ValueError):
                            checkpoints._verify_directory_at(binding, name, identity)
                    finally:
                        os.close(descriptor)
                        original.rmdir(); displaced.rename(original)
                root.rename(Path(directory) / "owned")
                root.mkdir()
                with self.assertRaises(ValueError):
                    binding.verify()

    def test_descriptor_read_rejects_record_replaced_between_open_and_use(self) -> None:
        context = _context()
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            Exp425CheckpointStore.create(root, _provenance(context), context)
            target = root / "datasets" / "record.json"
            target.write_bytes(b"{}")
            original_read = checkpoints.os.read
            replaced = False

            def replace_then_read(descriptor: int, size: int) -> bytes:
                nonlocal replaced
                if not replaced:
                    replaced = True
                    target.replace(root / "datasets" / "replacement.json")
                    target.write_bytes(b"{}")
                return original_read(descriptor, size)

            with checkpoints._locked_root(root, exclusive=False) as binding:
                descriptor, _ = checkpoints._open_directory_at(binding, "datasets")
                try:
                    with patch.object(checkpoints.os, "read", side_effect=replace_then_read):
                        with self.assertRaises(ValueError):
                            checkpoints._read_file("record.json", dir_fd=descriptor)
                finally:
                    os.close(descriptor)

    def test_import_promotion_revalidates_after_each_durable_phase_and_resumes_two_records_once(self) -> None:
        """A new source authority alone may resume every durable A→E boundary."""
        context = _context()
        provenance = _provenance(context)
        datasets = context.datasets[:2]
        batch_identity = "sealed-exp424-import"
        source_attestation = _digest("a")

        def imported_record(dataset: str, index: int) -> ImportedRecordEnvelope:
            evidence = {"dataset": dataset, "source_groups": [index]}
            return ImportedRecordEnvelope(
                dataset=dataset,
                result_payload={"status": "abstained", "source_index": index},
                group_evidence=SourceProvenGroupEvidence(
                    evidence, sha256(canonical_json_bytes(evidence)).hexdigest()
                ),
                source_root_sha256=sha256(f"root:{dataset}".encode()).hexdigest(),
                source_progress_sha256=sha256(f"progress:{dataset}".encode()).hexdigest(),
                source_record_sha256=sha256(f"record:{dataset}".encode()).hexdigest(),
                source_result_sha256=sha256(f"result:{dataset}".encode()).hexdigest(),
                compatibility_attestation_sha256=source_attestation,
                elapsed_seconds=float(index),
                reason="sealed local exp424 import",
            )

        records = tuple(imported_record(dataset, index) for index, dataset in enumerate(datasets))

        def stage_digest(store: Exp425CheckpointStore, record: ImportedRecordEnvelope) -> str:
            return sha256(canonical_json_bytes(checkpoints._import_record_body(store.run_context_sha256, record))).hexdigest()

        def final_attestation(store: Exp425CheckpointStore) -> str:
            staged = {record.dataset: stage_digest(store, record) for record in records}
            return sha256(canonical_json_bytes({
                "batch_identity": batch_identity,
                "record_digests": dict(sorted(staged.items())),
                "attestation_sha256": source_attestation,
            })).hexdigest()

        def begin_and_stage(store: Exp425CheckpointStore, authority: object, nonce: str) -> object:
            batch = store._begin_import_batch(capabilities._mint(
                authority, kind="import_begin", run_identity=provenance.run_identity,
                batch_identity=batch_identity, attestation_sha256=source_attestation,
                process_nonce=nonce,
            ), batch_identity)
            for record in records:
                store._stage_imported(capabilities._mint(
                    authority, kind="import_stage", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, dataset=record.dataset,
                    source_sha256=record.source_record_sha256,
                    attestation_sha256=source_attestation, process_nonce=nonce,
                    record_sha256=stage_digest(store, record),
                ), batch, record)
            return batch

        def promote(store: Exp425CheckpointStore, authority: object, nonce: str, batch: object) -> None:
            store._promote_import_batch(capabilities._mint(
                authority, kind="import_promote", run_identity=provenance.run_identity,
                batch_identity=batch_identity, attestation_sha256=source_attestation,
                process_nonce=nonce, final_attestation_sha256=final_attestation(store),
            ), batch)

        def file_bytes(root: Path) -> dict[str, bytes]:
            return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}

        class SimulatedProcessDeath(RuntimeError):
            pass

        for phase in ("intent_durable", "record_durable", "progress_durable", "intent_removed"):
            with self.subTest(phase=phase), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                store = Exp425CheckpointStore.create(root, provenance, context)
                old_authority, old_nonce = object(), "old-nonce"
                capabilities._register_authority(old_authority, "exp424_import")
                try:
                    old_batch = begin_and_stage(store, old_authority, old_nonce)
                    with patch.object(checkpoints, "_after_import_phase", side_effect=lambda reached: (
                        (_ for _ in ()).throw(SimulatedProcessDeath(reached)) if reached == phase else None
                    )):
                        with self.assertRaisesRegex(SimulatedProcessDeath, phase):
                            promote(store, old_authority, old_nonce, old_batch)
                finally:
                    capabilities._close_authority(old_authority)

                before_open = file_bytes(root)
                reopened = Exp425CheckpointStore.open(root, provenance, context)
                self.assertEqual(file_bytes(root), before_open)
                self.assertEqual(
                    reopened.import_revalidation_required,
                    ((batch_identity, final_attestation(reopened)),),
                )

                new_authority, new_nonce = object(), "new-nonce"
                capabilities._register_authority(new_authority, "exp424_import")
                try:
                    new_batch = begin_and_stage(reopened, new_authority, new_nonce)
                    promote(reopened, new_authority, new_nonce, new_batch)
                finally:
                    capabilities._close_authority(new_authority)
                expected_digests = {record.dataset: stage_digest(reopened, record) for record in records}
                self.assertEqual(dict(reopened.completed_record_digests()), expected_digests)
                self.assertEqual(json.loads((root / "progress.json").read_bytes())["record_digests"], expected_digests)
                self.assertEqual(set(path.name for path in (root / "datasets").iterdir()), {
                    checkpoints.record_filename(record.dataset) for record in records
                })
                self.assertEqual(list((root / "transactions").iterdir()), [])
                self.assertEqual(list((root / "import_batches").iterdir()), [])

    def test_invalid_import_capabilities_leave_the_checkpoint_unchanged(self) -> None:
        """Wrong source, closed, batch, and final-attestation tokens cannot mutate state."""
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            authority = object()
            capabilities._register_authority(authority, "exp424_import")
            record = ImportedRecordEnvelope(
                context.datasets[0], {"status": "abstained"},
                SourceProvenGroupEvidence({"source_groups": []}, _digest("b")),
                _digest("c"), _digest("d"), _digest("e"), _digest("f"), _digest("a"),
                0.0, "sealed local exp424 import",
            )
            batch_identity, attestation, nonce = "capability-batch", _digest("a"), "nonce"
            try:
                batch = store._begin_import_batch(capabilities._mint(
                    authority, kind="import_begin", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, attestation_sha256=attestation, process_nonce=nonce,
                ), batch_identity)
                digest = sha256(canonical_json_bytes(checkpoints._import_record_body(store.run_context_sha256, record))).hexdigest()

                def unchanged_after(action: object) -> None:
                    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                    with self.assertRaises(ValueError):
                        action()
                    self.assertEqual(
                        {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}, before
                    )

                unchanged_after(lambda: store._stage_imported(object(), batch, record))
                wrong_source = capabilities._mint(authority, kind="import_stage", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, dataset=record.dataset, source_sha256=_digest("0"),
                    attestation_sha256=attestation, process_nonce=nonce, record_sha256=digest)
                unchanged_after(lambda: store._stage_imported(wrong_source, batch, record))
                wrong_batch = capabilities._mint(authority, kind="import_stage", run_identity=provenance.run_identity,
                    batch_identity="other-batch", dataset=record.dataset, source_sha256=record.source_record_sha256,
                    attestation_sha256=attestation, process_nonce=nonce, record_sha256=digest)
                unchanged_after(lambda: store._stage_imported(wrong_batch, batch, record))
                store._stage_imported(capabilities._mint(authority, kind="import_stage", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, dataset=record.dataset, source_sha256=record.source_record_sha256,
                    attestation_sha256=attestation, process_nonce=nonce, record_sha256=digest), batch, record)
                final = sha256(canonical_json_bytes({
                    "batch_identity": batch_identity,
                    "record_digests": {record.dataset: digest},
                    "attestation_sha256": attestation,
                })).hexdigest()
                wrong_final = capabilities._mint(authority, kind="import_promote", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, attestation_sha256=attestation, process_nonce=nonce,
                    final_attestation_sha256=_digest("0"))
                unchanged_after(lambda: store._promote_import_batch(wrong_final, batch))
                closed = capabilities._mint(authority, kind="import_promote", run_identity=provenance.run_identity,
                    batch_identity=batch_identity, attestation_sha256=attestation, process_nonce=nonce,
                    final_attestation_sha256=final)
                capabilities._close_authority(authority)
                unchanged_after(lambda: store._promote_import_batch(closed, batch))
            finally:
                capabilities._close_authority(authority)

    def test_import_cleanup_crashes_are_revalidated_and_resumed_once(self) -> None:
        """Every cleanup boundary leaves durable authority-gated restart state."""
        class SimulatedProcessDeath(RuntimeError):
            pass

        context = _context()
        provenance = _provenance(context)
        batch_identity, attestation = "cleanup-batch", _digest("a")
        records = tuple(_import_record_for(dataset, index, attestation) for index, dataset in enumerate(context.datasets[:2]))
        cases = (("cleanup_tombstone_durable", 1), ("cleanup_staged_unlinked", 1), ("cleanup_staged_unlinked", 2), ("cleanup_metadata_removed", 1), ("cleanup_directory_removed", 1), ("cleanup_directory_fsynced", 1))
        for phase, occurrence in cases:
            with self.subTest(phase=phase, occurrence=occurrence), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoints"
                store = Exp425CheckpointStore.create(root, provenance, context)
                old_authority = object()
                capabilities._register_authority(old_authority, "exp424_import")
                try:
                    batch = _begin_and_stage_imports(store, provenance, old_authority, "old-nonce", batch_identity, attestation, records)
                    seen = 0

                    def crash_after(reached: str) -> None:
                        nonlocal seen
                        if reached == phase:
                            seen += 1
                            if seen == occurrence:
                                raise SimulatedProcessDeath(reached)

                    with patch.object(checkpoints, "_after_import_cleanup_phase", side_effect=crash_after):
                        with self.assertRaisesRegex(SimulatedProcessDeath, phase):
                            _promote_imports(store, provenance, old_authority, "old-nonce", batch_identity, attestation, records, batch)
                finally:
                    capabilities._close_authority(old_authority)
                before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
                reopened = Exp425CheckpointStore.open(root, provenance, context)
                self.assertEqual({str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)
                self.assertEqual(reopened.import_revalidation_required, ((batch_identity, _import_final_attestation(reopened, batch_identity, attestation, records)),))
                new_authority = object()
                capabilities._register_authority(new_authority, "exp424_import")
                try:
                    resumed = _begin_and_stage_imports(reopened, provenance, new_authority, "new-nonce", batch_identity, attestation, records)
                    _promote_imports(reopened, provenance, new_authority, "new-nonce", batch_identity, attestation, records, resumed)
                finally:
                    capabilities._close_authority(new_authority)
                self.assertEqual(list((root / "transactions").iterdir()), [])
                self.assertEqual(list((root / "import_batches").iterdir()), [])
                self.assertEqual(len(reopened.completed_record_digests()), 2)

    def test_cleanup_rejects_a_replaced_batch_without_touching_it(self) -> None:
        class SimulatedProcessDeath(RuntimeError):
            pass

        context = _context()
        provenance = _provenance(context)
        batch_identity, attestation = "replaced-cleanup-batch", _digest("a")
        records = (_import_record_for(context.datasets[0], 0, attestation),)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            authority = object()
            capabilities._register_authority(authority, "exp424_import")
            try:
                batch = _begin_and_stage_imports(store, provenance, authority, "old-nonce", batch_identity, attestation, records)
                with patch.object(checkpoints, "_after_import_cleanup_phase", side_effect=lambda phase: (_ for _ in ()).throw(SimulatedProcessDeath()) if phase == "cleanup_tombstone_durable" else None):
                    with self.assertRaises(SimulatedProcessDeath):
                        _promote_imports(store, provenance, authority, "old-nonce", batch_identity, attestation, records, batch)
            finally:
                capabilities._close_authority(authority)
            batch_path = root / "import_batches" / batch_identity
            batch_path.rename(Path(directory) / "displaced-batch")
            batch_path.mkdir()
            foreign = batch_path / checkpoints.record_filename(context.datasets[0])
            foreign.write_bytes(b"unrelated")
            before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "replaced"):
                Exp425CheckpointStore.inspect_read_only(root)
            self.assertEqual({str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}, before)

    def test_import_failures_do_not_leak_descriptors_or_follow_replaced_root(self) -> None:
        context = _context()
        provenance = _provenance(context)
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            authority = object()
            capabilities._register_authority(authority, "exp424_import")
            try:
                baseline = len(os.listdir("/dev/fd"))
                for index in range(10):
                    batch = _begin_and_stage_imports(store, provenance, authority, f"nonce-{index}", f"fd-batch-{index}", _digest("a"), ())
                    token = capabilities._mint(authority, kind="import_stage", run_identity=provenance.run_identity,
                        batch_identity=batch.identity, dataset=context.datasets[0], source_sha256=_digest("b"),
                        attestation_sha256=_digest("a"), process_nonce=f"nonce-{index}", record_sha256=_digest("c"))
                    with patch.object(checkpoints, "_open_batch_at", side_effect=ValueError("adversarial open failure")):
                        with self.assertRaises(ValueError):
                            store._stage_imported(token, batch, _import_record_for(context.datasets[0], 0, _digest("a")))
                self.assertEqual(len(os.listdir("/dev/fd")), baseline)
            finally:
                capabilities._close_authority(authority)
            displaced = Path(directory) / "displaced"
            root.rename(displaced)
            root.mkdir()
            authority = object()
            capabilities._register_authority(authority, "exp424_import")
            try:
                with self.assertRaisesRegex(ValueError, "store root identity"):
                    _begin_and_stage_imports(store, provenance, authority, "replaced", "replacement-batch", _digest("a"), ())
            finally:
                capabilities._close_authority(authority)
            self.assertEqual(list(root.iterdir()), [])


class CheckpointBenchmarkTests(unittest.TestCase):
    """The compact format must avoid revalidating the same frozen context 25 times."""

    def test_shared_context_is_compact_and_resumes_faster_than_repeated_context(self) -> None:
        metrics = _measure_shared_context_checkpoint_benchmark()
        self.assertGreaterEqual(metrics.byte_reduction_fraction, 0.80)
        self.assertGreaterEqual(metrics.speedup, 2.0)
        self.assertEqual(metrics.exp425_candidate_coordinates, metrics.old_candidate_coordinates)
        self.assertEqual(metrics.exp425_result_evidence_digests, metrics.old_result_evidence_digests)
        self.assertEqual(len(metrics.exp425_candidate_coordinates), 25)
        self.assertEqual(metrics.completed_record_count, 25)
