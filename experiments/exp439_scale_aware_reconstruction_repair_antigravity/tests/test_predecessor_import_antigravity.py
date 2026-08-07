from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.checkpoint_antigravity import (
    Exp439CheckpointStore,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.config_antigravity import (
    FROZEN_CONFIG,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.predecessor_import_antigravity import (
    EXP438_PREFIX_ROOT_ANTIGRAVITY,
    _validate_exp438_prefix_source_antigravity,
    validate_exp438_prefix_antigravity,
)


class Exp439PredecessorImportAntigravityTests(unittest.TestCase):
    def setUp(self) -> None:
        if not EXP438_PREFIX_ROOT_ANTIGRAVITY.is_dir():
            self.skipTest("the immutable Exp438 predecessor root is unavailable")

    def _copy_predecessor(self, parent: Path) -> Path:
        destination = parent / "exp438_mirror_antigravity"
        shutil.copytree(EXP438_PREFIX_ROOT_ANTIGRAVITY, destination)
        return destination

    def _exp439_context(self, datasets: tuple[str, ...]) -> dict[str, object]:
        return {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "run_identity": FROZEN_CONFIG.run_identity,
            "code_commit": "a" * 40,
            "database_sha256": FROZEN_CONFIG.database_sha256,
            "datasets": list(datasets),
            "backend": "cpu",
        }

    def test_exact_failed_prefix_is_read_only_and_returns_89_records(self) -> None:
        before = {
            path.relative_to(EXP438_PREFIX_ROOT_ANTIGRAVITY): sha256(path.read_bytes()).hexdigest()
            for path in EXP438_PREFIX_ROOT_ANTIGRAVITY.rglob("*")
            if path.is_file()
        }

        imported = validate_exp438_prefix_antigravity(EXP438_PREFIX_ROOT_ANTIGRAVITY)

        self.assertEqual(len(imported.records), 89)
        self.assertEqual(imported.records[-1].dataset, "BeetleFly_normal_1")
        self.assertEqual(imported.datasets[89], "BeetleFly_normal_2")
        self.assertEqual(imported.prediction_label_access_count, 0)
        self.assertEqual(imported.records[0].sequence_index, 0)
        self.assertEqual(
            sha256(imported.records[0].source_bytes).hexdigest(),
            imported.records[0].source_sha256,
        )
        self.assertEqual(
            imported.records[0].result.dataset,
            imported.records[0].dataset,
        )
        self.assertIsNotNone(imported.records[0].dataset_metadata)
        after = {
            path.relative_to(EXP438_PREFIX_ROOT_ANTIGRAVITY): sha256(path.read_bytes()).hexdigest()
            for path in EXP438_PREFIX_ROOT_ANTIGRAVITY.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)

    def test_one_altered_record_byte_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._copy_predecessor(Path(temporary))
            record = sorted((root / "checkpoint_antigravity/records_antigravity").iterdir())[0]
            raw = bytearray(record.read_bytes())
            raw[20] ^= 1
            record.write_bytes(raw)
            with self.assertRaises(ValueError):
                _validate_exp438_prefix_source_antigravity(root)

    def test_missing_reordered_or_extra_record_is_rejected(self) -> None:
        mutations = ("missing", "reordered", "extra")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = self._copy_predecessor(Path(temporary))
                records = root / "checkpoint_antigravity/records_antigravity"
                paths = sorted(records.iterdir())
                if mutation == "missing":
                    paths[0].unlink()
                elif mutation == "reordered":
                    first = paths[0].read_bytes()
                    second = paths[1].read_bytes()
                    paths[0].write_bytes(second)
                    paths[1].write_bytes(first)
                else:
                    (records / "0089_extra_antigravity.json").write_bytes(paths[-1].read_bytes())
                with self.assertRaises(ValueError):
                    _validate_exp438_prefix_source_antigravity(root)

    def test_wrong_commit_or_nonzero_label_access_is_rejected(self) -> None:
        for field, value in (("reviewed_commit", "0" * 40), ("prediction_label_access_count", 1)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = self._copy_predecessor(Path(temporary))
                readiness = root / "readiness_antigravity.json"
                payload = json.loads(readiness.read_bytes())
                payload[field] = value
                readiness.write_bytes(canonical_json_bytes(payload))
                with self.assertRaises(ValueError):
                    _validate_exp438_prefix_source_antigravity(root)

    def test_imported_checkpoint_appears_only_after_all_89_records_commit(self) -> None:
        imported = validate_exp438_prefix_antigravity(EXP438_PREFIX_ROOT_ANTIGRAVITY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"

            store = Exp439CheckpointStore.create_with_exp438_prefix_antigravity(
                root,
                self._exp439_context(imported.datasets),
                imported,
            )

            self.assertEqual(len(store.committed_datasets), 89)
            self.assertEqual(store.committed_datasets[-1], "BeetleFly_normal_1")
            manifest = json.loads((root / "import_manifest_antigravity.json").read_bytes())
            self.assertEqual(manifest["record_count"], 89)
            self.assertEqual(manifest["next_dataset"], "BeetleFly_normal_2")
            self.assertEqual(len(manifest["records"]), 89)
            state = json.loads((root / "state_antigravity.json").read_bytes())
            self.assertEqual(
                [item["destination_sha256"] for item in manifest["records"]],
                [item["sha256"] for item in state["committed"]],
            )
            self.assertFalse(tuple(Path(temporary).glob(".antigravity_*")))

    def test_invalid_import_creates_no_checkpoint_root(self) -> None:
        imported = validate_exp438_prefix_antigravity(EXP438_PREFIX_ROOT_ANTIGRAVITY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            with self.assertRaises(ValueError):
                Exp439CheckpointStore.create_with_exp438_prefix_antigravity(
                    root,
                    self._exp439_context(imported.datasets),
                    object(),
                )
            self.assertFalse(root.exists())
            self.assertFalse(tuple(Path(temporary).glob(".antigravity_*")))

    def test_import_manifest_tampering_is_rejected_by_bound_hash(self) -> None:
        imported = validate_exp438_prefix_antigravity(EXP438_PREFIX_ROOT_ANTIGRAVITY)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint_antigravity"
            store = Exp439CheckpointStore.create_with_exp438_prefix_antigravity(
                root,
                self._exp439_context(imported.datasets),
                imported,
            )
            manifest_path = root / "import_manifest_antigravity.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["records"][0]["source_sha256"] = "0" * 64
            manifest_path.write_bytes(canonical_json_bytes(manifest))

            with self.assertRaisesRegex(ValueError, "import manifest"):
                Exp439CheckpointStore.open_existing(
                    root,
                    store.context_payload,
                    mutable=False,
                )


if __name__ == "__main__":
    unittest.main()
