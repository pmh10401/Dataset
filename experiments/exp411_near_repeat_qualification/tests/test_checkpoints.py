from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp411_near_repeat_qualification.checkpoints import (
    CheckpointProvenance,
    CheckpointStore,
)
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)
from experiments.exp411_near_repeat_qualification.perturbations import (
    perturbation_types,
)


def _row(dataset: str = "Alpha_normal_0") -> DatasetQualification:
    types = perturbation_types()
    return DatasetQualification(
        dataset=dataset,
        family=dataset.split("_normal_", 1)[0],
        identity_group_count=6,
        normal_total=6,
        normal_abstained=0,
        normal_near={"0.01": 1, "0.05": 2, "0.10": 3},
        synthetic_total={"0.01": 42, "0.05": 42, "0.10": 42},
        synthetic_near={"0.01": 0, "0.05": 1, "0.10": 2},
        perturbation_total={
            key: {name: 6 for name in types}
            for key in ("0.01", "0.05", "0.10")
        },
        perturbation_near={
            key: {
                name: int(
                    (key == "0.05" and name == "spike")
                    or (
                        key == "0.10"
                        and name in {"spike", "drift"}
                    )
                )
                for name in types
            }
            for key in ("0.01", "0.05", "0.10")
        },
    )


def _audit() -> DatasetBackendAudit:
    return DatasetBackendAudit(
        backend="numba_cpu",
        sampled_pairs=((0, 1), (1, 2)),
        max_abs_error=1e-12,
        fallback_reason=None,
    )


def _provenance() -> CheckpointProvenance:
    return CheckpointProvenance(
        git_commit="1" * 40,
        database_snapshot={"db": "2" * 64},
        population_manifest_sha256="3" * 64,
        frozen_config_sha256="4" * 64,
        perturbation_contract_sha256="5" * 64,
        acceleration_contract_sha256="6" * 64,
    )


class CheckpointTests(unittest.TestCase):
    def test_dataset_checkpoint_round_trips_for_resume(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )

            store.write_dataset(
                _row(),
                _audit(),
                "7" * 64,
                1.25,
            )

            resumed = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )
            self.assertEqual(
                resumed.load_completed(),
                {"Alpha_normal_0": _row()},
            )
            self.assertEqual(
                tuple(root.glob("datasets/*.tmp")),
                (),
            )

    def test_provenance_drift_is_rejected_without_changing_checkpoint(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)
            before = tuple(
                (path.name, path.read_bytes())
                for path in sorted(root.rglob("*"))
                if path.is_file()
            )

            with self.assertRaisesRegex(ValueError, "provenance"):
                CheckpointStore(
                    root,
                    replace(_provenance(), git_commit="f" * 40),
                    ("Alpha_normal_0",),
                )

            after = tuple(
                (path.name, path.read_bytes())
                for path in sorted(root.rglob("*"))
                if path.is_file()
            )
            self.assertEqual(after, before)

    def test_corrupt_dataset_checkpoint_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)
            checkpoint = next((root / "datasets").glob("*.json"))
            checkpoint.write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "checkpoint"):
                store.load_completed()

    def test_duplicate_dataset_checkpoint_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)
            checkpoint = next((root / "datasets").glob("*.json"))
            duplicate = checkpoint.with_name("duplicate.json")
            duplicate.write_bytes(checkpoint.read_bytes())

            with self.assertRaisesRegex(ValueError, "duplicated"):
                store.load_completed()

    def test_incomplete_coverage_cannot_be_completed(self) -> None:
        with TemporaryDirectory() as raw:
            store = CheckpointStore(
                Path(raw) / "checkpoint",
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)

            with self.assertRaisesRegex(ValueError, "coverage"):
                store.require_complete(expected_families=2)

    def test_failed_rewrite_preserves_existing_dataset_checkpoint(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0",),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)
            checkpoint = next((root / "datasets").glob("*.json"))
            before = checkpoint.read_bytes()

            with self.assertRaisesRegex(ValueError, "already exists"):
                store.write_dataset(
                    replace(_row(), normal_total=7),
                    _audit(),
                    "8" * 64,
                    2.5,
                )

            self.assertEqual(checkpoint.read_bytes(), before)

    def test_progress_reports_only_completion_count_and_current_dataset(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            store.write_dataset(_row(), _audit(), "7" * 64, 1.25)

            store.write_progress(current_dataset="Beta_normal_0")

            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(progress),
                {"completed_datasets", "current_dataset", "expected_datasets"},
            )
            self.assertEqual(progress["completed_datasets"], 1)
            self.assertEqual(progress["expected_datasets"], 2)
            self.assertEqual(progress["current_dataset"], "Beta_normal_0")


if __name__ == "__main__":
    unittest.main()
