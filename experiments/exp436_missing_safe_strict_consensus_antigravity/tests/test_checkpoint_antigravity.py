"""Durability contract tests for the Exp436 ordered checkpoint."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import fcntl
import json
import multiprocessing
import os
import stat
import threading
from hashlib import sha256
from unittest.mock import patch
from experiments.exp403_408.artifact_io import canonical_json_bytes

import experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity as checkpoint
from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import (
    Exp436CheckpointStore,
    Exp436RunContext,
    context_fingerprint,
    _name,
    _result_payload,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import BackendDecision, Exp436DatasetResult, InstanceScoreEvidence


def _context() -> dict[str, object]:
    from hashlib import sha256
    from experiments.exp403_408.artifact_io import canonical_json_bytes
    datasets = [f"D{i:04d}" for i in range(1117)]
    manifest = {name: [[2, 16]] for name in datasets}
    lanes = {name: {"path": f"/tmp/{name}_antigravity", "sha256": "a" * 64} for name in ("exp291_prediction", "exp291_manifest", "exp292_prediction", "exp292_manifest")}
    decision = BackendDecision.cpu_fallback().to_payload()
    lane_mappings = {"hard":{},"standard":{},"priority":{},"automatic_normal":{}}
    from dataclasses import asdict
    from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG
    config = json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    payload = {"schema":"exp436.run-context.v1","protocol_versions":{"context":"v1","checkpoint":"v1","readiness":"v1"},"experiment_id":"Exp436","run_identity":"exp436-missing-safe-strict-consensus-antigravity-20260806","reviewed_commit":"b"*40,"readiness_path":"/tmp/readiness_antigravity.json","readiness_artifact_sha256":"a"*64,"backend_decision":decision,"backend_decision_sha256":sha256(canonical_json_bytes(decision)).hexdigest(),"database_path":"/tmp/db_antigravity","database_sha256":"a"*64,"lane_artifacts":lanes,"lane_mappings":lane_mappings,"lane_snapshot_sha256":sha256(canonical_json_bytes({"artifacts":lanes,"mappings":lane_mappings,"datasets":datasets})).hexdigest(),"datasets":datasets,"dataset_order_sha256":sha256(canonical_json_bytes(datasets)).hexdigest(),"test_identity_manifest":manifest,"test_manifest_sha256":sha256(canonical_json_bytes(manifest)).hexdigest(),"frozen_config":config,"frozen_config_sha256":sha256(canonical_json_bytes(config)).hexdigest(),"python_implementation":"CPython","python_version":"3","python_executable":"/tmp/python","numpy_version":"2","platform":"x","machine":"x","multiprocessing_start_method":"spawn","logical_cpus":12,"cpu_workers":11,"selected_framework_provenance":{"framework":"cpu","adopted":False},"backend_protocol":"paired-synchronized-v1","selected_batch_size":None,"output_path":"/tmp/output_antigravity","checkpoint_path":"/tmp/checkpoint_antigravity"}
    payload["context_sha256"] = sha256(canonical_json_bytes(payload)).hexdigest()
    return Exp436RunContext.from_payload(payload)


def _result(dataset: str, index: int) -> Exp436DatasetResult:
    evidence = InstanceScoreEvidence.fixture(dataset=dataset, instance_index=index, length=16)
    return Exp436DatasetResult(dataset, "scored", (index,), (), (evidence,))


def _commit_after_inspection_antigravity(root_text: str, started: object, completed: object, result: object) -> None:
    """Spawn-safe writer used only to prove the root flock blocks mutation."""
    try:
        started.set()
        store = Exp436CheckpointStore.open_existing(Path(root_text), _context())
        store.commit(_result("D0000", 2))
        result.put("committed")
    except BaseException as error:
        result.put(f"{type(error).__name__}:{error}")
    finally:
        completed.set()


class CheckpointTests(unittest.TestCase):
    def test_record_persists_exact_nonlabel_dataset_metadata_for_runtime_replay(self) -> None:
        directory, root, store = self._make_store()
        try:
            metadata = {"train_count": 4, "test_count": 1, "train_length_histogram": {"16": 4}, "test_length_histogram": {"16": 1}}
            store.commit(_result("D0000", 2), dataset_metadata=metadata)
            self.assertEqual(dict(store.load_record_metadata("D0000")), metadata)
        finally:
            directory.cleanup()

    def test_missing_record_metadata_cannot_be_used_for_runtime_replay(self) -> None:
        directory, root, store = self._make_store()
        try:
            store.commit(_result("D0000", 2))
            with self.assertRaisesRegex(ValueError, "metadata"):
                store.load_record_metadata("D0000")
        finally:
            directory.cleanup()
    def _bad_context(self, key: str, value: object) -> None:
        payload = _context().to_payload(); payload[key] = value
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)

    def test_context_rejects_unknown_key(self) -> None: self._bad_context("unknown", 1)
    def test_context_rejects_wrong_commit(self) -> None: self._bad_context("reviewed_commit", "bad")
    def test_context_rejects_wrong_population(self) -> None: self._bad_context("datasets", ["D0000"])
    def test_context_rejects_duplicate_manifest_index(self) -> None:
        payload = _context().to_payload(); payload["test_identity_manifest"]["D0000"] = [[2,16],[2,17]]
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)
    def test_context_rejects_relative_path(self) -> None: self._bad_context("database_path", "relative")
    def test_context_rejects_bad_backend_digest(self) -> None: self._bad_context("backend_decision_sha256", "b" * 64)
    def test_context_rejects_bad_context_digest(self) -> None: self._bad_context("context_sha256", "b" * 64)
    def test_context_roundtrip_is_exact(self) -> None: self.assertEqual(_context().to_payload(), Exp436RunContext.from_payload(_context().to_payload()).to_payload())
    def test_context_rejects_missing_key(self) -> None:
        payload = _context().to_payload(); payload.pop("machine")
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)
    def test_context_rejects_unordered_population(self) -> None:
        payload = _context().to_payload(); payload["datasets"][0],payload["datasets"][1]=payload["datasets"][1],payload["datasets"][0]
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)
    def test_context_rejects_missing_manifest_dataset(self) -> None:
        payload = _context().to_payload(); payload["test_identity_manifest"].pop("D0000")
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)
    def test_context_rejects_bad_lane_artifact(self) -> None:
        payload = _context().to_payload(); payload["lane_artifacts"]["exp291_manifest"]["path"]="x"
        with self.assertRaises(ValueError): Exp436RunContext.from_payload(payload)
    def test_checkpoint_rejects_arbitrary_context(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): Exp436CheckpointStore.create(Path(directory)/"checkpoint_antigravity", {"datasets":[]})

    def test_dataset_digest_rejects_resigned_outer_context(self) -> None:
        payload=_context().to_payload(); payload["datasets"][0]="X"; payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"population|dataset order"): Exp436RunContext.from_payload(payload)
    def test_manifest_digest_rejects_resigned_outer_context(self) -> None:
        payload=_context().to_payload(); payload["test_identity_manifest"]["D0000"]=[[3,16]]; payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"manifest"): Exp436RunContext.from_payload(payload)
    def test_lane_digest_rejects_resigned_outer_context(self) -> None:
        payload=_context().to_payload(); payload["lane_mappings"]["hard"]={"D0000":[2]}; payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"lane snapshot"): Exp436RunContext.from_payload(payload)
    def test_frozen_config_rejects_resigned_outer_context(self) -> None:
        payload=_context().to_payload(); payload["frozen_config"]["cpu_workers"]=1; payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"frozen config"): Exp436RunContext.from_payload(payload)
    def test_decision_rejects_resigned_outer_context(self) -> None:
        payload=_context().to_payload(); payload["backend_decision"]={}; payload["backend_decision_sha256"]=sha256(canonical_json_bytes({})).hexdigest(); payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"backend decision"): Exp436RunContext.from_payload(payload)
    def test_lane_overlap_rejects(self) -> None:
        payload=_context().to_payload(); payload["lane_mappings"]={"hard":{"D0000":[2]},"standard":{"D0000":[2]},"priority":{},"automatic_normal":{}}; payload["lane_snapshot_sha256"]=sha256(canonical_json_bytes({"artifacts":payload["lane_artifacts"],"mappings":payload["lane_mappings"],"datasets":payload["datasets"]})).hexdigest(); payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"overlap"): Exp436RunContext.from_payload(payload)
    def test_protocol_schema_rejects(self) -> None: self._bad_context("protocol_versions", {"context":"v1"})
    def test_selected_provenance_rejects(self) -> None: self._bad_context("selected_framework_provenance", {"framework":"mlx","adopted":False})
    def test_worker_topology_rejects(self) -> None: self._bad_context("cpu_workers", 1)
    def test_backend_protocol_rejects(self) -> None: self._bad_context("backend_protocol", "wrong")
    def test_lane_index_type_rejects(self) -> None:
        payload=_context().to_payload(); payload["lane_mappings"]={"hard":{"D0000":["2"]},"standard":{},"priority":{},"automatic_normal":{}}; payload["lane_snapshot_sha256"]=sha256(canonical_json_bytes({"artifacts":payload["lane_artifacts"],"mappings":payload["lane_mappings"],"datasets":payload["datasets"]})).hexdigest(); payload["context_sha256"]=sha256(canonical_json_bytes({k:v for k,v in payload.items() if k!="context_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError,"lane mapping"): Exp436RunContext.from_payload(payload)
    def test_record_set_counter_schema_exists(self) -> None:
        with TemporaryDirectory() as directory:
            root=Path(directory)/"checkpoint_antigravity"; Exp436CheckpointStore.create(root,_context()); payload=json.loads((root/"record_set_antigravity.json").read_bytes()); self.assertEqual(payload["candidate_count"],0); self.assertEqual(payload["completed_datasets"],[])

    def _crash_state(self, stage: str):
        directory=TemporaryDirectory(); root=Path(directory.name)/"checkpoint_antigravity"; store=Exp436CheckpointStore.create(root,_context())
        def fail(point: str) -> None:
            if point == stage: raise RuntimeError(point)
        with self.assertRaisesRegex(RuntimeError, stage): store.commit(_result("D0000",2),_failpoint=fail)
        return directory,root
    def test_failpoint_after_intent_reports_and_repairs(self) -> None:
        directory,root=self._crash_state("after_intent_fsync")
        try:
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).reconciliation_reasons,("intent_without_record",)); self.assertEqual(Exp436CheckpointStore.open_existing(root,_context()).inspect().completed,0)
        finally: directory.cleanup()
    def test_failpoint_after_record_reports_and_advances(self) -> None:
        directory,root=self._crash_state("after_record_fsync")
        try:
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).reconciliation_reasons,("intent_with_record",)); self.assertEqual(Exp436CheckpointStore.open_existing(root,_context()).inspect().completed,1)
        finally: directory.cleanup()
    def test_failpoint_after_set_reports_stale_and_repairs(self) -> None:
        directory,root=self._crash_state("after_record_set_fsync_before_intent_remove")
        try:
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).reconciliation_reasons,("stale_intent_after_record_set",)); self.assertEqual(Exp436CheckpointStore.open_existing(root,_context()).inspect().completed,1)
        finally: directory.cleanup()
    def test_create_commits_only_ordered_prefix_and_rejects_101st_before_intent(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp436CheckpointStore.create(root, _context())
            with self.assertRaisesRegex(ValueError, "prefix"):
                store.commit(_result("D0001", 2))
            store.commit(_result("D0000", 2))
            self.assertEqual(store.inspect().completed, 1)
            self.assertEqual(list((root / "intents_antigravity").iterdir()), [])

    def test_inspect_is_byte_preserving_and_resume_reconciles_intent_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint_antigravity"
            Exp436CheckpointStore.create(root, _context())
            intent = root / "intents_antigravity" / (sha256(b"D0000").hexdigest() + ".json")
            # A valid intent-only first crash state is read-only corruption
            # until mutable resume explicitly reconciles it.
            intent.write_text(json.dumps({"schema": "exp436.commit-intent.v1", "dataset": "D0000", "record_sha256": "a" * 64}, sort_keys=True, separators=(",", ":")))
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            inspection = Exp436CheckpointStore.inspect_read_only(root)
            self.assertTrue(inspection.reconciliation_required)
            self.assertEqual(inspection.reconciliation_reasons, ("intent_without_record",))
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            Exp436CheckpointStore.open_existing(root, _context())
            self.assertEqual(list((root / "intents_antigravity").iterdir()), [])

    def _snapshot(self, root: Path) -> tuple[tuple[str, str, bytes | str | None], ...]:
        entries: list[tuple[str, str, bytes | str | None]] = []
        for path in sorted(root.rglob("*")):
            info = os.lstat(path)
            relative = str(path.relative_to(root))
            if stat.S_ISLNK(info.st_mode):
                entries.append((relative, "symlink", os.readlink(path)))
            elif stat.S_ISREG(info.st_mode):
                entries.append((relative, "file", path.read_bytes()))
            elif stat.S_ISDIR(info.st_mode):
                entries.append((relative, "directory", None))
            else:
                entries.append((relative, "other", None))
        return tuple(entries)

    def _assert_rejected_pristine(self, root: Path) -> None:
        before = self._snapshot(root)
        with self.assertRaises(ValueError):
            Exp436CheckpointStore.inspect_read_only(root)
        self.assertEqual(self._snapshot(root), before)
        with self.assertRaises(ValueError):
            Exp436CheckpointStore.open_existing(root, _context())
        self.assertEqual(self._snapshot(root), before)

    def _make_store(self) -> tuple[TemporaryDirectory[str], Path, Exp436CheckpointStore]:
        directory = TemporaryDirectory()
        root = Path(directory.name) / "checkpoint_antigravity"
        return directory, root, Exp436CheckpointStore.create(root, _context())

    def _intent_bytes(self, dataset: str, digest: str) -> bytes:
        return canonical_json_bytes({"schema": "exp436.commit-intent.v1", "dataset": dataset, "record_sha256": digest})

    def _record_bytes(self, dataset: str, sequence_index: int) -> tuple[bytes, str]:
        raw = canonical_json_bytes(_result_payload(_result(dataset, 2), sequence_index))
        return raw, sha256(raw).hexdigest()

    def _valid_temp(self, root: Path, records: dict[str, str]) -> Path:
        results = tuple(_result(dataset, 2) for dataset in records)
        raw = canonical_json_bytes(Exp436CheckpointStore._set_payload_static(context_fingerprint(_context()), tuple(records), records, results))
        path = root / (".record_set_antigravity.json." + "a" * 32 + ".tmp")
        path.write_bytes(raw)
        return path

    def test_valid_current_record_set_temp_reports_then_mutably_removes(self) -> None:
        directory, root, _ = self._make_store()
        try:
            temporary = self._valid_temp(root, {})
            before = self._snapshot(root)
            inspection = Exp436CheckpointStore.inspect_read_only(root)
            self.assertEqual(inspection.reconciliation_reasons, ("record_set_temp",))
            self.assertEqual(self._snapshot(root), before)
            Exp436CheckpointStore.open_existing(root, _context())
            self.assertFalse(temporary.exists())
            self.assertEqual(Exp436CheckpointStore.open_read_only(root).inspect().completed, 0)
        finally:
            directory.cleanup()

    def test_valid_prospective_temp_reports_both_reasons_and_converges(self) -> None:
        directory, root, store = self._make_store()
        try:
            def fail(point: str) -> None:
                if point == "after_record_fsync":
                    raise RuntimeError(point)
            with self.assertRaisesRegex(RuntimeError, "after_record_fsync"):
                store.commit(_result("D0000", 2), _failpoint=fail)
            record = root / "records_antigravity" / _name("D0000")
            self._valid_temp(root, {"D0000": sha256(record.read_bytes()).hexdigest()})
            inspection = Exp436CheckpointStore.inspect_read_only(root)
            self.assertEqual(inspection.reconciliation_reasons, ("intent_with_record", "record_set_temp"))
            recovered = Exp436CheckpointStore.open_existing(root, _context())
            self.assertEqual(recovered.inspect().completed, 1)
            self.assertEqual(list((root / "intents_antigravity").iterdir()), [])
            self.assertEqual([path.name for path in root.iterdir() if path.name.startswith(".record_set")], [])
        finally:
            directory.cleanup()

    def test_malformed_or_noncanonical_temp_rejects_without_mutation(self) -> None:
        for kind in ("malformed", "noncanonical"):
            with self.subTest(kind=kind):
                directory, root, _ = self._make_store()
                try:
                    temporary = self._valid_temp(root, {})
                    if kind == "malformed":
                        temporary.write_bytes(canonical_json_bytes({"schema": "exp436.record-set.v1"}))
                    else:
                        temporary.write_bytes(temporary.read_bytes() + b"\n")
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_multiple_temps_reject_without_mutation(self) -> None:
        directory, root, _ = self._make_store()
        try:
            first = self._valid_temp(root, {})
            second = root / (".record_set_antigravity.json." + "b" * 32 + ".tmp")
            second.write_bytes(first.read_bytes())
            self._assert_rejected_pristine(root)
        finally:
            directory.cleanup()

    def test_unrelated_well_formed_temp_rejects_without_mutation(self) -> None:
        directory, root, _ = self._make_store()
        try:
            raw, digest = self._record_bytes("D0000", 0)
            (root / "records_antigravity" / _name("D0000")).write_bytes(raw)
            (root / "intents_antigravity" / _name("D0000")).write_bytes(self._intent_bytes("D0000", digest))
            _, other_digest = self._record_bytes("D0001", 1)
            self._valid_temp(root, {"D0001": other_digest})
            self._assert_rejected_pristine(root)
        finally:
            directory.cleanup()

    def test_unknown_root_entry_rejects_without_mutation(self) -> None:
        directory, root, _ = self._make_store()
        try:
            (root / ".hidden-antigravity").write_bytes(b"authority surprise")
            self._assert_rejected_pristine(root)
        finally:
            directory.cleanup()

    def test_multiple_intents_reject_without_mutation(self) -> None:
        directory, root, store = self._make_store()
        try:
            def fail(point: str) -> None:
                if point == "after_intent_fsync":
                    raise RuntimeError(point)
            with self.assertRaises(RuntimeError):
                store.commit(_result("D0000", 2), _failpoint=fail)
            _, future_digest = self._record_bytes("D0001", 1)
            (root / "intents_antigravity" / _name("D0001")).write_bytes(self._intent_bytes("D0001", future_digest))
            self._assert_rejected_pristine(root)
        finally:
            directory.cleanup()

    def test_future_and_past_nonstale_intents_reject_without_mutation(self) -> None:
        for state in ("future", "past"):
            with self.subTest(state=state):
                directory, root, store = self._make_store()
                try:
                    if state == "future":
                        _, digest = self._record_bytes("D0001", 1)
                        dataset = "D0001"
                    else:
                        store.commit(_result("D0000", 2))
                        dataset, digest = "D0000", "b" * 64
                    (root / "intents_antigravity" / _name(dataset)).write_bytes(self._intent_bytes(dataset, digest))
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_orphan_record_rejects_without_mutation(self) -> None:
        directory, root, _ = self._make_store()
        try:
            raw, _ = self._record_bytes("D0000", 0)
            (root / "records_antigravity" / _name("D0000")).write_bytes(raw)
            self._assert_rejected_pristine(root)
        finally:
            directory.cleanup()

    def test_intent_record_digest_and_sequence_mismatches_reject_without_mutation(self) -> None:
        for state in ("digest", "sequence"):
            with self.subTest(state=state):
                directory, root, store = self._make_store()
                try:
                    def fail(point: str) -> None:
                        if point == "after_record_fsync":
                            raise RuntimeError(point)
                    with self.assertRaises(RuntimeError):
                        store.commit(_result("D0000", 2), _failpoint=fail)
                    record = root / "records_antigravity" / _name("D0000")
                    intent = root / "intents_antigravity" / _name("D0000")
                    if state == "digest":
                        intent.write_bytes(self._intent_bytes("D0000", "b" * 64))
                    else:
                        raw, digest = self._record_bytes("D0000", 1)
                        record.write_bytes(raw)
                        intent.write_bytes(self._intent_bytes("D0000", digest))
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_unknown_record_and_intent_entries_reject_without_mutation(self) -> None:
        for state in ("record", "intent"):
            with self.subTest(state=state):
                directory, root, _ = self._make_store()
                try:
                    raw, digest = self._record_bytes("D0001", 1)
                    if state == "record":
                        (root / "records_antigravity" / _name("D0001")).write_bytes(raw)
                    else:
                        (root / "intents_antigravity" / _name("D0001")).write_bytes(self._intent_bytes("D0001", digest))
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_symlink_authorities_reject_without_following_or_mutation(self) -> None:
        for authority in ("run_context", "record_set", "record", "intent", "temp"):
            with self.subTest(authority=authority):
                directory, root, store = self._make_store()
                try:
                    if authority == "record":
                        store.commit(_result("D0000", 2))
                        path = root / "records_antigravity" / _name("D0000")
                    elif authority == "intent":
                        def fail(point: str) -> None:
                            if point == "after_intent_fsync":
                                raise RuntimeError(point)
                        with self.assertRaises(RuntimeError):
                            store.commit(_result("D0000", 2), _failpoint=fail)
                        path = root / "intents_antigravity" / _name("D0000")
                    elif authority == "temp":
                        path = self._valid_temp(root, {})
                    else:
                        path = root / ("run_context_antigravity.json" if authority == "run_context" else "record_set_antigravity.json")
                    target = path.with_name(path.name + ".target")
                    path.rename(target)
                    path.symlink_to(target.name)
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_directory_authorities_reject_without_mutation(self) -> None:
        for authority in ("run_context", "record_set", "record", "intent", "temp"):
            with self.subTest(authority=authority):
                directory, root, store = self._make_store()
                try:
                    if authority == "record":
                        store.commit(_result("D0000", 2))
                        path = root / "records_antigravity" / _name("D0000")
                    elif authority == "intent":
                        def fail(point: str) -> None:
                            if point == "after_intent_fsync":
                                raise RuntimeError(point)
                        with self.assertRaises(RuntimeError):
                            store.commit(_result("D0000", 2), _failpoint=fail)
                        path = root / "intents_antigravity" / _name("D0000")
                    elif authority == "temp":
                        path = self._valid_temp(root, {})
                    else:
                        path = root / ("run_context_antigravity.json" if authority == "run_context" else "record_set_antigravity.json")
                    path.unlink()
                    path.mkdir()
                    self._assert_rejected_pristine(root)
                finally:
                    directory.cleanup()

    def test_inspect_holds_one_shared_lock_for_every_authority_read(self) -> None:
        directory, root, _ = self._make_store()
        try:
            calls: list[int] = []
            original_flock = checkpoint.fcntl.flock
            original_regular = checkpoint._regular_bytes
            def traced_flock(descriptor: int, operation: int) -> object:
                calls.append(operation)
                return original_flock(descriptor, operation)
            def observed_regular(path: Path, label: str) -> bytes:
                self.assertEqual(calls, [fcntl.LOCK_SH])
                return original_regular(path, label)
            with patch.object(checkpoint.fcntl, "flock", side_effect=traced_flock), patch.object(checkpoint, "_regular_bytes", side_effect=observed_regular):
                inspection = Exp436CheckpointStore.inspect_read_only(root)
            self.assertEqual(inspection.completed, 0)
            self.assertEqual(calls, [fcntl.LOCK_SH, fcntl.LOCK_UN])
        finally:
            directory.cleanup()

    def test_inspect_shared_lock_blocks_concurrent_writer_until_release(self) -> None:
        directory, root, _ = self._make_store()
        try:
            entered, release = threading.Event(), threading.Event()
            inspected: list[object] = []
            original_regular = checkpoint._regular_bytes
            def paused_regular(path: Path, label: str) -> bytes:
                if label == "run context" and not entered.is_set():
                    entered.set()
                    self.assertTrue(release.wait(10))
                return original_regular(path, label)
            def inspect() -> None:
                inspected.append(Exp436CheckpointStore.inspect_read_only(root))
            with patch.object(checkpoint, "_regular_bytes", side_effect=paused_regular):
                reader = threading.Thread(target=inspect)
                reader.start()
                self.assertTrue(entered.wait(10))
                ctx = multiprocessing.get_context("spawn")
                started, completed, result = ctx.Event(), ctx.Event(), ctx.Queue()
                writer = ctx.Process(target=_commit_after_inspection_antigravity, args=(str(root), started, completed, result))
                writer.start()
                self.assertTrue(started.wait(10))
                self.assertFalse(completed.wait(1))
                release.set()
                reader.join(10)
                writer.join(10)
            self.assertFalse(reader.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertEqual(result.get(timeout=1), "committed")
            self.assertEqual(inspected[0].completed, 0)
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).completed, 1)
        finally:
            directory.cleanup()

    def test_mutable_open_uses_one_exclusive_lock_across_scan_repair_rescan(self) -> None:
        directory, root, store = self._make_store()
        try:
            def fail(point: str) -> None:
                if point == "after_record_fsync":
                    raise RuntimeError(point)
            with self.assertRaises(RuntimeError):
                store.commit(_result("D0000", 2), _failpoint=fail)
            calls: list[int] = []
            original_flock = checkpoint.fcntl.flock
            def traced_flock(descriptor: int, operation: int) -> object:
                calls.append(operation)
                return original_flock(descriptor, operation)
            with patch.object(checkpoint.fcntl, "flock", side_effect=traced_flock):
                recovered = Exp436CheckpointStore.open_existing(root, _context())
            self.assertEqual(calls, [fcntl.LOCK_EX, fcntl.LOCK_UN])
            self.assertEqual(recovered.inspect().completed, 1)
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).completed, 1)
        finally:
            directory.cleanup()

    def test_commit_uses_one_exclusive_lock_per_coherent_idempotent_call(self) -> None:
        directory, root, store = self._make_store()
        try:
            result = _result("D0000", 2)
            original_flock = checkpoint.fcntl.flock
            for expected_completed in (1, 1):
                calls: list[int] = []
                def traced_flock(descriptor: int, operation: int) -> object:
                    calls.append(operation)
                    return original_flock(descriptor, operation)
                with patch.object(checkpoint.fcntl, "flock", side_effect=traced_flock):
                    store.commit(result)
                self.assertEqual(calls, [fcntl.LOCK_EX, fcntl.LOCK_UN])
                self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).completed, expected_completed)
        finally:
            directory.cleanup()

    def test_context_and_inspection_projections_are_deeply_immutable(self) -> None:
        directory, root, _ = self._make_store()
        try:
            context = Exp436CheckpointStore.read_context_projection(root)
            inspection = Exp436CheckpointStore.inspect_read_only(root)
            with self.assertRaises(TypeError):
                context["lane_artifacts"]["exp291_manifest"]["path"] = "/tmp/changed"
            with self.assertRaises(AttributeError):
                context["datasets"].append("changed")
            with self.assertRaises(TypeError):
                inspection.record_digests["D0000"] = "b" * 64
            self.assertEqual(Exp436CheckpointStore.read_context_projection(root)["datasets"][0], "D0000")
            self.assertEqual(Exp436CheckpointStore.inspect_read_only(root).completed, 0)
        finally:
            directory.cleanup()
