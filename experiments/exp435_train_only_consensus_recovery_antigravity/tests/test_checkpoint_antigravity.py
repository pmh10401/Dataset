"""TDD checks for the Exp435 canonical checkpoint store."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.checkpoint_antigravity import (
    Exp435CheckpointStore,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    Exp435DatasetResult,
    InstanceScoreEvidence,
)


def _context() -> dict[str, object]:
    return {
        "schema": "fixture.context.v1",
        "datasets": ["Demo", "Other"],
        "test_sizes": {"Demo": 2, "Other": 2},
        "database_sha256": "d" * 64,
        "code_commit": "f" * 40,
        "runtime_versions": {"python": "test"},
    }


def _result(dataset: str = "Demo") -> Exp435DatasetResult:
    evidence = InstanceScoreEvidence.fixture(dataset=dataset, instance_index=0)
    return Exp435DatasetResult(
        dataset=dataset,
        status="scored",
        test_size=2,
        evidence=(evidence,),
    )


class Exp435CheckpointTests(unittest.TestCase):
    def test_create_commit_reopen_and_record_set_digest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp435CheckpointStore.create(root, _context())
            digest = store.commit(_result())
            reopened = Exp435CheckpointStore.open_existing(root, _context())
            self.assertEqual(reopened.completed_digests()["Demo"], digest)
            self.assertEqual(reopened.load_record("Demo"), _result())
            self.assertEqual(reopened.inspect().completed, 1)
            self.assertTrue(reopened.inspect().record_set_sha256)

    def test_record_commit_survives_missing_event_and_progress(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp435CheckpointStore.create(root, _context())
            store.commit(_result())
            reopened = Exp435CheckpointStore.open_existing(root, _context())
            recovered = reopened.recover_runtime_accounting()
            self.assertEqual(recovered.completed, 1)
            self.assertEqual(recovered.missing_events, ("Demo",))

    def test_context_mismatch_refuses_resume(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp435CheckpointStore.create(root, _context())
            altered = dict(_context())
            altered["code_commit"] = "a" * 40
            with self.assertRaisesRegex(ValueError, "context fingerprint"):
                Exp435CheckpointStore.open_existing(root, altered)

    def test_no_clobber_and_unknown_file_rejection(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp435CheckpointStore.create(root, _context())
            with self.assertRaisesRegex(ValueError, "exists"):
                Exp435CheckpointStore.create(root, _context())
            (root / "unknown_antigravity.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                Exp435CheckpointStore.open_existing(root, _context())

    def test_exact_test_size_is_required_before_commit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp435CheckpointStore.create(root, _context())
            with self.assertRaisesRegex(ValueError, "TEST size"):
                store.commit(Exp435DatasetResult.fixture(dataset="Demo", test_size=3))
            self.assertEqual(tuple((root / "records_antigravity").iterdir()), ())

    def test_mutable_open_reconciles_internal_temp_but_unknown_temp_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp435CheckpointStore.create(root, _context())
            internal = root / (".record_set_antigravity." + "a" * 32 + ".tmp")
            internal.write_bytes(b"partial")
            Exp435CheckpointStore.open_existing(root, _context())
            self.assertFalse(internal.exists())
            unknown = root / ".record_set_antigravity.unknown.tmp"
            unknown.write_bytes(b"partial")
            with self.assertRaisesRegex(ValueError, "inventory"):
                Exp435CheckpointStore.open_existing(root, _context())

    def test_read_only_inspection_rejects_temp_without_mutating_tree(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp435CheckpointStore.create(root, _context())
            internal = root / (".record_set_antigravity." + "b" * 32 + ".tmp")
            internal.write_bytes(b"partial")
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "temporary"):
                Exp435CheckpointStore.inspect_read_only(root)
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_read_only_inspection_rejects_pending_intent_without_reconciling(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp435CheckpointStore.create(root, _context())
            dataset = "Demo"
            intent = root / "intents_antigravity" / (sha256(dataset.encode()).hexdigest() + ".json")
            intent.write_text(
                json.dumps({
                    "schema": "exp435.commit-intent.v1",
                    "dataset": dataset,
                    "record_sha256": "0" * 64,
                }, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            before = intent.read_bytes()
            with self.assertRaisesRegex(ValueError, "pending commit intent"):
                Exp435CheckpointStore.inspect_read_only(root)
            self.assertEqual(intent.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
