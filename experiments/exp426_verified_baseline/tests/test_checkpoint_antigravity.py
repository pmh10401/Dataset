"""Crash-safe checkpoint tests for Exp426."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp426_verified_baseline.checkpoint_antigravity import Exp426CheckpointStore
from experiments.exp426_verified_baseline.contracts_antigravity import (
    Exp426DatasetRecord,
    Exp426Provenance,
)


class Exp426CheckpointTests(unittest.TestCase):
    @staticmethod
    def canonical(value: object) -> bytes:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @staticmethod
    def record_filename(dataset: str) -> str:
        return sha256(dataset.encode("utf-8")).hexdigest() + ".json"

    @staticmethod
    def provenance() -> Exp426Provenance:
        return Exp426Provenance(
            run_identity="exp426-run", git_commit="1" * 40,
            database_sha256="2" * 64, source_root_sha256="3" * 64,
            source_progress_sha256="4" * 64, source_record_set_sha256="5" * 64,
            lane_artifact_sha256={
                "hard": "6" * 64, "standard": "7" * 64,
                "priority": "8" * 64, "automatic_normal": "9" * 64,
            },
            fold_contract_sha256="a" * 64,
            expected_datasets=("A_normal_1", "B_normal_1"), workers=12,
        )

    @staticmethod
    def context() -> dict[str, object]:
        return {
            "schema": "exp426.frozen-context.v1",
            "database_sha256": "2" * 64,
            "datasets": ["A_normal_1", "B_normal_1"],
            "test_sizes": {"A_normal_1": 3, "B_normal_1": 2},
            "lane_coordinates": [],
            "code_commit": "1" * 40,
            "runtime_versions": {"numpy": "test", "python": "3.14-test"},
        }

    @staticmethod
    def imported() -> Exp426DatasetRecord:
        return Exp426DatasetRecord(
            "A_normal_1", "exp424_import", "scored", (("A_normal_1", 1),),
            "b" * 64, "c" * 64, None, None, "verified_exp424_nonterminal",
        )

    @staticmethod
    def abstention() -> Exp426DatasetRecord:
        return Exp426DatasetRecord(
            "B_normal_1", "verified_train_abstention", "abstained", (),
            "d" * 64, "e" * 64, "f" * 64, "0" * 64,
            "oof_identity_insufficient",
        )

    def test_create_commit_reopen_and_read_only_inspect(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp426CheckpointStore.create(root, self.provenance(), self.context())
            store.commit(self.imported())
            store.commit(self.abstention())
            self.assertEqual(store.completed_digests(), {
                "A_normal_1": self.imported().sha256,
                "B_normal_1": self.abstention().sha256,
            })
            reopened = Exp426CheckpointStore.open(root, self.provenance(), self.context())
            self.assertEqual(reopened.load_record("A_normal_1"), self.imported())
            before = {p.relative_to(root): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                      for p in root.rglob("*") if p.is_file()}
            inspection = Exp426CheckpointStore.inspect_read_only(root, expected_run_identity="exp426-run")
            after = {p.relative_to(root): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                     for p in root.rglob("*") if p.is_file()}
            self.assertEqual((inspection.completed, inspection.total), (2, 2))
            self.assertEqual(before, after)

    def test_create_and_commit_are_no_clobber(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp426CheckpointStore.create(root, self.provenance(), self.context())
            with self.assertRaisesRegex(ValueError, "exists"):
                Exp426CheckpointStore.create(root, self.provenance(), self.context())
            store.commit(self.imported())
            store.commit(self.imported())
            changed = Exp426DatasetRecord(
                "A_normal_1", "exp424_import", "scored", (),
                "b" * 64, "c" * 64, None, None, "verified_exp424_nonterminal",
            )
            with self.assertRaisesRegex(ValueError, "different"):
                store.commit(changed)

    def test_open_rejects_context_mismatch_and_corrupt_progress(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp426CheckpointStore.create(root, self.provenance(), self.context())
            changed = dict(self.context())
            changed["test_sizes"] = {"A_normal_1": 4, "B_normal_1": 2}
            with self.assertRaisesRegex(ValueError, "context"):
                Exp426CheckpointStore.open(root, self.provenance(), changed)
            (root / "progress.json").write_text(json.dumps({"broken": True}))
            with self.assertRaises(ValueError):
                Exp426CheckpointStore.inspect_read_only(root)

    def test_context_requires_nonempty_string_runtime_versions(self) -> None:
        with TemporaryDirectory() as directory:
            context = self.context()
            context["runtime_versions"] = {}
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                Exp426CheckpointStore.create(
                    Path(directory) / "checkpoint", self.provenance(), context,
                )

    def test_record_outside_universe_and_lane_overlap_are_rejected_before_write(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            context = self.context()
            context["lane_coordinates"] = [["A_normal_1", 1]]
            store = Exp426CheckpointStore.create(root, self.provenance(), context)
            with self.assertRaisesRegex(ValueError, "lane"):
                store.commit(self.imported())
            outside = Exp426DatasetRecord(
                "Z_normal_1", "exp424_import", "scored", (),
                "b" * 64, "c" * 64, None, None, "verified_exp424_nonterminal",
            )
            with self.assertRaisesRegex(ValueError, "universe"):
                store.commit(outside)
            self.assertEqual(list((root / "datasets").iterdir()), [])

    def test_reopen_reconciles_every_durable_commit_boundary(self) -> None:
        record = self.imported()
        record_raw = self.canonical(record.scientific_payload())
        record_digest = sha256(record_raw).hexdigest()
        intent_raw = self.canonical({
            "schema": "exp426.commit-intent.v1",
            "dataset": record.dataset,
            "record_sha256": record_digest,
        })
        progress_raw = self.canonical({
            "schema": "exp426.checkpoint-progress.v1",
            "run_identity": "exp426-run",
            "completed": 1,
            "total": 2,
            "record_digests": {record.dataset: record_digest},
            "current_dataset": None,
        })
        for boundary, write_dataset, write_progress, expected_completed in (
            ("intent", False, False, 0),
            ("dataset", True, False, 1),
            ("progress", True, True, 1),
        ):
            with self.subTest(boundary=boundary), TemporaryDirectory() as directory:
                root = Path(directory) / "checkpoint"
                Exp426CheckpointStore.create(root, self.provenance(), self.context())
                name = self.record_filename(record.dataset)
                (root / "intents" / name).write_bytes(intent_raw)
                if write_dataset:
                    (root / "datasets" / name).write_bytes(record_raw)
                if write_progress:
                    (root / "progress.json").write_bytes(progress_raw)

                reopened = Exp426CheckpointStore.open(
                    root, self.provenance(), self.context()
                )

                self.assertEqual(len(reopened.completed_digests()), expected_completed)
                self.assertEqual(list((root / "intents").iterdir()), [])
                if expected_completed:
                    self.assertEqual(reopened.load_record(record.dataset), record)

    def test_stale_writable_handle_cannot_lose_a_completed_record(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            first = Exp426CheckpointStore.create(root, self.provenance(), self.context())
            stale = Exp426CheckpointStore.open(root, self.provenance(), self.context())

            first.commit(self.imported())
            stale.commit(self.abstention())

            inspection = Exp426CheckpointStore.inspect_read_only(root)
            self.assertEqual(inspection.completed, 2)
            self.assertEqual(set(inspection.record_digests), {"A_normal_1", "B_normal_1"})

    def test_open_rejects_unknown_and_symlink_inventory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            Exp426CheckpointStore.create(root, self.provenance(), self.context())
            unknown = root / "unknown.json"
            unknown.write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "inventory"):
                Exp426CheckpointStore.inspect_read_only(root)
            unknown.unlink()

            symlink = root / "intents" / self.record_filename("A_normal_1")
            symlink.symlink_to(root / "progress.json")
            with self.assertRaisesRegex(ValueError, "intent inventory"):
                Exp426CheckpointStore.inspect_read_only(root)

    def test_open_rejects_non_lowercase_progress_digest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            store = Exp426CheckpointStore.create(root, self.provenance(), self.context())
            store.commit(self.imported())
            progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            progress["record_digests"]["A_normal_1"] = self.imported().sha256.upper()
            (root / "progress.json").write_bytes(self.canonical(progress))

            with self.assertRaisesRegex(ValueError, "lowercase"):
                Exp426CheckpointStore.inspect_read_only(root)


if __name__ == "__main__":
    unittest.main()
