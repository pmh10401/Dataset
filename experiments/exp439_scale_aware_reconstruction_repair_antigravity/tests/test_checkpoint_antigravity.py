import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.checkpoint_antigravity import (
    Exp439CheckpointStore,
    context_fingerprint_antigravity,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import (
    Exp439DatasetResult,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
)


def _context() -> dict[str, object]:
    return {
        "experiment_id": "Exp439",
        "run_identity": "exp439-frequency-complexity-evidence-antigravity-20260807",
        "code_commit": "a" * 40,
        "database_sha256": "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
        "datasets": ["A", "B"],
        "backend": "cpu",
    }


def _result(dataset: str, index: int) -> Exp439DatasetResult:
    channels = tuple(
        InstanceChannelEvidence(name, 2.0, 1.0, True, True, None)
        for name in ("high_frequency", "low_frequency", "amplitude_complexity")
    )
    return Exp439DatasetResult(
        dataset,
        (InstanceScoreEvidence(dataset, index, 16, channels, True),),
        (),
        (),
    )


def _metadata() -> dict[str, object]:
    return {
        "train_count": 4,
        "test_count": 2,
        "train_length_histogram": {"16": 4},
        "test_length_histogram": {"16": 2},
        "train_value_count": 64,
        "test_value_count": 32,
        "train_bytes": 256,
        "test_bytes": 128,
    }


class Exp439CheckpointAntigravityTests(unittest.TestCase):
    def test_create_commit_reopen_and_ordered_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create(root, _context())
            self.assertEqual(store.context_hash, context_fingerprint_antigravity(_context()))
            store.commit(_result("A", 0))
            store.mark_interrupted()
            reopened = Exp439CheckpointStore.open_existing(root, _context(), mutable=True)
            self.assertEqual(reopened.committed_datasets, ("A",))
            reopened.commit(_result("B", 1))
            reopened.mark_complete()
            replay = Exp439CheckpointStore.open_existing(root, _context(), mutable=False)
            self.assertEqual(tuple(result.dataset for result in replay.results()), ("A", "B"))
            self.assertEqual(replay.status, "complete")
            self.assertFalse(tuple(root.rglob("*tmp*")))

    def test_out_of_order_commit_is_rejected_before_record_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create(root, _context())
            with self.assertRaisesRegex(ValueError, "next frozen dataset"):
                store.commit(_result("B", 1))
            self.assertEqual(store.committed_datasets, ())
            self.assertEqual(tuple((root / "records_antigravity").iterdir()), ())

    def test_record_set_fingerprint_and_dataset_metadata_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create(root, _context())
            initial = json.loads((root / "state_antigravity.json").read_text())
            store.commit(_result("A", 0), dataset_metadata=_metadata())
            committed = json.loads((root / "state_antigravity.json").read_text())
            self.assertNotEqual(initial["record_set_hash"], committed["record_set_hash"])
            reopened = Exp439CheckpointStore.open_existing(root, _context(), mutable=False)
            self.assertEqual(reopened.metadata_for_dataset("A"), _metadata())
            self.assertIsNone(reopened.metadata_for_dataset("B"))

    def test_existing_root_is_never_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            root.mkdir()
            marker = root / "marker_antigravity.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                Exp439CheckpointStore.create(root, _context())
            self.assertEqual(marker.read_text(), "keep")

    def test_failed_root_is_immutable_and_cross_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create(root, _context())
            store.mark_failed("synthetic_failure")
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "failed checkpoint is immutable"):
                Exp439CheckpointStore.open_existing(root, _context(), mutable=True)
            forged = dict(_context(), experiment_id="Exp437")
            with self.assertRaisesRegex(ValueError, "Exp439 identity"):
                Exp439CheckpointStore.open_existing(root, forged, mutable=False)
            after = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_orphan_or_tampered_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create(root, _context())
            store.commit(_result("A", 0))
            orphan = root / "records_antigravity" / "9999_orphan_antigravity.json"
            orphan.write_text("{}")
            with self.assertRaisesRegex(ValueError, "record inventory"):
                Exp439CheckpointStore.open_existing(root, _context(), mutable=False)

    def test_context_rejects_forbidden_prediction_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = dict(_context(), nested={"label": 1})
            with self.assertRaisesRegex(ValueError, "forbidden prediction field"):
                Exp439CheckpointStore.create(Path(temporary) / "checkpoint_antigravity", context)


if __name__ == "__main__":
    unittest.main()
