"""Compact shared-context checkpoint tests for Exp425."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

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
            result_payload = {"effective_result": {"status": "abstained"}, "native_trace_body": trace_body}
            body = {"schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native",
                "run_context_sha256": store.run_context_sha256, "result_payload": result_payload, "group_evidence": {"groups": []},
                "trace_sha256": trace, "model_hashes": {}, "calibration_hashes": {}, "backend_audit_sha256": values["backend_audit_sha256"],
                "elapsed_seconds": 0.0, "reason": "synthetic"}
            record = NativeRecordEnvelope(dataset, result_payload, {"groups": []}, trace_sha256=trace,
                record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(), model_hashes={}, calibration_hashes={}, elapsed_seconds=0.0,
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


class CheckpointBenchmarkTests(unittest.TestCase):
    """The compact format must avoid revalidating the same frozen context 25 times."""

    def test_shared_context_is_compact_and_resumes_faster_than_repeated_context(self) -> None:
        context = _context()
        provenance = _provenance(context)
        datasets = context.datasets[:25]

        def record_for(dataset: str, index: int, store: Exp425CheckpointStore) -> NativeRecordEnvelope:
            digest = lambda label: sha256(f"{dataset}:{label}".encode()).hexdigest()
            values = {name: digest(name) for name in ("train_identity_sha256", "eligibility_sha256", "processed_groups_sha256", "model_evidence_sha256", "frozen_train_contract_sha256", "scoring_evidence_sha256", "result_evidence_sha256", "backend_audit_sha256")}
            trace_body = {"schema": "exp425.native-trace.v1", "run_identity": provenance.run_identity, "dataset": dataset, "run_context_sha256": store.run_context_sha256, **values, "device": "mps"}
            trace = sha256(canonical_json_bytes(trace_body)).hexdigest()
            result = {"effective_result": {"candidate_coordinates": [[dataset, index]], "result_evidence_sha256": values["result_evidence_sha256"], "status": "abstained"}, "native_trace_body": trace_body}
            body = {"schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native", "run_context_sha256": store.run_context_sha256, "result_payload": result, "group_evidence": {"eligibility_sha256": values["eligibility_sha256"]}, "trace_sha256": trace, "model_hashes": {"model": values["model_evidence_sha256"]}, "calibration_hashes": {"score": values["scoring_evidence_sha256"]}, "backend_audit_sha256": values["backend_audit_sha256"], "elapsed_seconds": float(index), "reason": "benchmark"}
            return NativeRecordEnvelope(dataset, result, {"eligibility_sha256": values["eligibility_sha256"]}, trace_sha256=trace, record_payload_sha256=sha256(canonical_json_bytes(body)).hexdigest(), model_hashes={"model": values["model_evidence_sha256"]}, calibration_hashes={"score": values["scoring_evidence_sha256"]}, elapsed_seconds=float(index), reason="benchmark", **values)

        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, provenance, context)
            authority = object(); capabilities._register_authority(authority, "exp425_native")
            try:
                records = [record_for(dataset, index, store) for index, dataset in enumerate(datasets)]
                for record in records:
                    token = capabilities._mint(authority, kind="native", run_identity=provenance.run_identity, dataset=record.dataset, context_sha256=store.run_context_sha256, trace_sha256=record.trace_sha256, record_sha256=record.record_payload_sha256, backend_sha256=record.backend_audit_sha256)
                    store._commit_native(token, record)
            finally:
                capabilities._close_authority(authority)
            repeated_records = [canonical_json_bytes({"schema": "exp424.repeated-context.v1", "context": _context_payload(context), "dataset": record.dataset, "result_payload": dict(record.result_payload), "group_evidence": dict(record.group_evidence), "model_hashes": dict(record.model_hashes), "calibration_hashes": dict(record.calibration_hashes), "backend_audit_sha256": record.backend_audit_sha256, "record_payload_sha256": record.record_payload_sha256}) for record in records]
            repeated_context_bytes = sum(map(len, repeated_records))
            exp425_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
            def old_cycle() -> list[list[object]]:
                payloads = [json.loads(raw) for raw in repeated_records]
                for raw, payload in zip(repeated_records, payloads, strict=True):
                    self.assertEqual(canonical_json_bytes(payload), raw)
                    self.assertEqual(_context_from_payload(payload["context"]), context)
                    self.assertEqual(sha256(canonical_json_bytes(payload["result_payload"]["native_trace_body"])).hexdigest(), sha256(canonical_json_bytes(payload["result_payload"]["native_trace_body"])).hexdigest())
                return [payload["result_payload"]["effective_result"]["candidate_coordinates"] for payload in payloads]
            def new_cycle() -> list[list[object]]:
                reopened = Exp425CheckpointStore.open(root, provenance, context)
                return [dict(record.result_payload)["effective_result"]["candidate_coordinates"] for record in reopened.iter_records()]
            old_cycle(); new_cycle()
            old_samples = []; new_samples = []
            for _ in range(5):
                started = time.perf_counter(); old_coordinates = old_cycle(); old_samples.append(time.perf_counter() - started)
                started = time.perf_counter(); exp425_coordinates = new_cycle(); new_samples.append(time.perf_counter() - started)
            old_resume_seconds = sorted(old_samples)[2]
            exp425_resume_seconds = sorted(new_samples)[2]
            self.assertLess(exp425_bytes, repeated_context_bytes * 0.20)
            self.assertLess(exp425_resume_seconds, old_resume_seconds / 2.0)
            self.assertEqual(exp425_coordinates, old_coordinates)
            self.assertEqual(len(store.completed_record_digests()), 25)
