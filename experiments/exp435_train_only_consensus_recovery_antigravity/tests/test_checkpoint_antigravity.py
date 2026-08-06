"""TDD checks for the Exp435 canonical checkpoint store."""

from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
