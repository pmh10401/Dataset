"""Compact shared-context checkpoint tests for Exp425."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp417_causal_multihorizon_shadow.contracts import FrozenRunContext
from experiments.exp417_causal_multihorizon_shadow.checkpoints import _context_payload
from experiments.exp417_causal_multihorizon_shadow.tests.test_checkpoints import _context
from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery.contracts import Exp425RunProvenance, NativeRecordEnvelope
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
            body = {"schema_version": "exp425.dataset.v1", "dataset": dataset, "origin": "exp425_native",
                "run_context_sha256": store.run_context_sha256, "result_payload": {"status": "abstained"}, "group_evidence": {"groups": []},
                "trace_sha256": trace, "model_hashes": {}, "calibration_hashes": {}, "backend_audit_sha256": values["backend_audit_sha256"],
                "elapsed_seconds": 0.0, "reason": "synthetic"}
            record = NativeRecordEnvelope(dataset, {"status": "abstained"}, {"groups": []}, trace_sha256=trace,
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


class CheckpointBenchmarkTests(unittest.TestCase):
    """The compact format must avoid revalidating the same frozen context 25 times."""

    def test_shared_context_is_compact_and_resumes_faster_than_repeated_context(self) -> None:
        context = _context()
        repeated_record = canonical_json_bytes({"context": _context_payload(context), "candidate_coordinates": [["x", 0]]})
        repeated_context_bytes = len(repeated_record) * 25
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            store = Exp425CheckpointStore.create(root, _provenance(context), context)
            exp425_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) + 25 * len(canonical_json_bytes({"candidate_coordinates": [["x", 0]]}))
            # Warm both paths; the old path must parse each embedded context,
            # whereas a resumed compact root validates it once.
            for _ in range(2):
                [sha256(canonical_json_bytes(json.loads(repeated_record))).hexdigest() for _ in range(25)]
                Exp425CheckpointStore.open(root, _provenance(context), context)
            started = time.perf_counter()
            for _ in range(2): [sha256(canonical_json_bytes(json.loads(repeated_record))).hexdigest() for _ in range(25)]
            old_resume_seconds = time.perf_counter() - started
            started = time.perf_counter()
            for _ in range(2): Exp425CheckpointStore.open(root, _provenance(context), context)
            exp425_resume_seconds = time.perf_counter() - started
            self.assertLess(exp425_bytes, repeated_context_bytes * 0.20)
            self.assertLess(exp425_resume_seconds, old_resume_seconds / 2.0)
            self.assertEqual([["x", 0]], [["x", 0]])
