from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)
from experiments.exp411_near_repeat_qualification.perturbations import (
    perturbation_types,
)
from experiments.exp414_missing_aware_near_repeat.checkpoints import (
    Exp414CheckpointProvenance,
    Exp414CheckpointStore,
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
                    or (key == "0.10" and name in {"spike", "drift"})
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


def _provenance() -> Exp414CheckpointProvenance:
    return Exp414CheckpointProvenance(
        git_commit="1" * 40,
        database_snapshot={"db": "2" * 64},
        population_manifest_sha256="3" * 64,
        frozen_config_sha256="4" * 64,
        perturbation_contract_sha256="5" * 64,
        acceleration_contract_sha256="6" * 64,
        missing_repair_contract_sha256="7" * 64,
        missing_inventory_sha256="8" * 64,
        qualification_contract_sha256="9" * 64,
        exp411_migration_source_sha256="a" * 64,
    )


def _write(
    store: Exp414CheckpointStore,
    row: DatasetQualification,
    *,
    execution_source: str = "exp414_computed",
) -> None:
    store.write_dataset(
        row,
        _audit(),
        raw_train_identity_sha256="1" * 64,
        repaired_train_identity_sha256="1" * 64,
        repair_audit_sha256="2" * 64,
        execution_source=execution_source,
        elapsed_seconds=0.25,
    )


class Exp414CheckpointTests(unittest.TestCase):
    def test_two_dataset_store_round_trips_record_and_counts_valid_records(self) -> None:
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            row = _row()

            _write(store, row)

            self.assertEqual(store.progress()["completed_datasets"], 1)
            self.assertEqual(
                store.load_dataset_record(row.dataset)["execution_source"],
                "exp414_computed",
            )
            self.assertTrue(store.has_dataset(row.dataset))
            self.assertFalse(store.has_dataset("Beta_normal_0"))
            record = store.load_dataset_record(row.dataset)
            self.assertEqual(
                set(record),
                {
                    "schema_version",
                    "dataset",
                    "raw_train_identity_sha256",
                    "repaired_train_identity_sha256",
                    "repair_audit_sha256",
                    "execution_source",
                    "qualification",
                    "backend_audit",
                    "elapsed_seconds",
                    "payload_sha256",
                },
            )

    def test_corrupt_payload_hash_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = Exp414CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            _write(store, _row())
            path = next((root / "datasets").glob("*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["payload_sha256"] = "0" * 64
            path.write_bytes(canonical_json_bytes(payload) + b"\n")

            with self.assertRaisesRegex(ValueError, "payload hash"):
                store.has_dataset("Alpha_normal_0")

    def test_foreign_dataset_record_fails_closed(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = Exp414CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            _write(store, _row())
            source = next((root / "datasets").glob("*.json"))
            payload = json.loads(source.read_text(encoding="utf-8"))
            payload["dataset"] = "Gamma_normal_0"
            payload["qualification"]["dataset"] = "Gamma_normal_0"
            payload["qualification"]["family"] = "Gamma"
            payload.pop("payload_sha256")
            payload["payload_sha256"] = sha256(
                canonical_json_bytes(payload)
            ).hexdigest()
            (root / "datasets" / "foreign.json").write_bytes(
                canonical_json_bytes(payload) + b"\n"
            )

            with self.assertRaisesRegex(ValueError, "foreign dataset"):
                store.progress()

    def test_boolean_counter_is_rejected_instead_of_coerced(self) -> None:
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            malformed = replace(
                _row(),
                normal_total=True,
                normal_near={"0.01": 0, "0.05": 0, "0.10": 0},
            )

            with self.assertRaisesRegex(ValueError, "counter"):
                _write(store, malformed)

    def test_wrong_root_provenance_is_rejected_without_mutation(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = Exp414CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            _write(store, _row())
            before = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            with self.assertRaisesRegex(ValueError, "provenance"):
                Exp414CheckpointStore(
                    root,
                    replace(
                        _provenance(),
                        missing_inventory_sha256="f" * 64,
                    ),
                    ("Alpha_normal_0", "Beta_normal_0"),
                )

            after = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(after, before)

    def test_invalid_execution_source_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )

            with self.assertRaisesRegex(ValueError, "execution source"):
                _write(store, _row(), execution_source="trusted_by_assumption")

    def test_duplicate_write_is_non_atomic_and_preserves_original(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = Exp414CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            _write(store, _row())
            path = next((root / "datasets").glob("*.json"))
            before = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "already exists"):
                _write(store, replace(_row(), normal_near={"0.01": 0, "0.05": 0, "0.10": 0}))

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(tuple((root / "datasets").glob("*.tmp")), ())

    def test_concurrent_duplicate_writes_publish_once_without_clobbering(self) -> None:
        with TemporaryDirectory() as raw:
            root = Path(raw) / "checkpoint"
            store = Exp414CheckpointStore(
                root,
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            rows = (
                _row(),
                replace(
                    _row(),
                    normal_near={"0.01": 0, "0.05": 0, "0.10": 0},
                ),
            )
            start = threading.Barrier(2)
            replace_boundary = threading.Barrier(2)
            original_replace = os.replace
            completed: list[int] = []
            errors: list[BaseException] = []

            def race_at_publish(source: object, destination: object) -> None:
                replace_boundary.wait(timeout=5.0)
                original_replace(source, destination)

            def write(index: int) -> None:
                try:
                    start.wait(timeout=5.0)
                    _write(store, rows[index])
                    completed.append(index)
                except BaseException as error:
                    errors.append(error)

            with patch(
                "experiments.exp414_missing_aware_near_repeat.checkpoints.os.replace",
                side_effect=race_at_publish,
            ):
                threads = tuple(
                    threading.Thread(target=write, args=(index,))
                    for index in range(2)
                )
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10.0)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ValueError)
            self.assertEqual(
                store.load_dataset_record("Alpha_normal_0")["qualification"],
                rows[completed[0]],
            )
            self.assertEqual(tuple((root / "datasets").glob("*.tmp")), ())

    def test_require_complete_requires_exact_dataset_and_family_coverage(self) -> None:
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            _write(store, _row())
            with self.assertRaisesRegex(ValueError, "dataset coverage"):
                store.require_complete(2)

            _write(store, _row("Beta_normal_0"))
            with self.assertRaisesRegex(ValueError, "family coverage"):
                store.require_complete(128)
            self.assertEqual(
                tuple(row.dataset for row in store.require_complete(2)),
                ("Alpha_normal_0", "Beta_normal_0"),
            )


if __name__ == "__main__":
    unittest.main()
