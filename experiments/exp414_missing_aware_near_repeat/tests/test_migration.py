from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.checkpoints import (
    CheckpointProvenance,
    CheckpointStore,
)
from experiments.exp411_near_repeat_qualification.config import (
    FROZEN_CONFIG as EXP411_FROZEN_CONFIG,
)
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    qualify_dataset,
)
from experiments.exp411_near_repeat_qualification.runner import (
    dataset_train_identity_sha256,
)
from experiments.exp414_missing_aware_near_repeat.checkpoints import (
    Exp414CheckpointProvenance,
    Exp414CheckpointStore,
)
from experiments.exp414_missing_aware_near_repeat.config import FROZEN_CONFIG
from experiments.exp414_missing_aware_near_repeat.contracts import PreparedDataset
from experiments.exp414_missing_aware_near_repeat.migration import (
    EXP411_MIGRATION_REPRESENTATIVES,
    migrate_if_eligible,
    open_exp411_checkpoint,
    verify_migration_representatives,
)
from experiments.exp414_missing_aware_near_repeat.repair import (
    build_dataset_repair_audit,
    repair_train_blob,
)


REPRESENTATIVES = (
    "ACSF1_normal_0",
    "CBF_normal_2",
    "DistalPhalanxTW_normal_8",
)
TARGET = "Alpha_normal_0"


def _prepared(
    dataset: str,
    *,
    repaired: bool = False,
    offset: float = 0.0,
) -> PreparedDataset:
    rows = []
    audits = []
    values = (
        (0.0, 1.0, 2.0),
        (2.0, 3.0, 4.0),
        (4.0, 5.0, 6.0),
    )
    for index, source in enumerate(values):
        array = np.asarray(
            tuple(value + offset for value in source),
            dtype="<f4",
        )
        if repaired and index == 1:
            array[1] = np.nan
        run, audit = repair_train_blob(dataset, index, array.tobytes())
        assert run is not None
        rows.append(run)
        audits.append(audit)
    repair_audit = build_dataset_repair_audit(dataset, rows, audits)
    return PreparedDataset(
        dataset=dataset,
        runs=tuple(rows),
        repair_audit=repair_audit,
    )


def _qualification(prepared: PreparedDataset) -> DatasetQualification:
    return qualify_dataset(
        dataset=prepared.dataset,
        family=prepared.dataset.split("_normal_", 1)[0],
        runs=prepared.runs,
        backend="numpy_cpu",
        accelerated=False,
        config=EXP411_FROZEN_CONFIG,
    )


def _backend_audit() -> DatasetBackendAudit:
    return DatasetBackendAudit(
        backend="numpy_cpu",
        sampled_pairs=(),
        max_abs_error=0.0,
        fallback_reason=None,
    )


def _source_provenance(
    *,
    git_commit: str = FROZEN_CONFIG.exp411_source_commit,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        git_commit=git_commit,
        database_snapshot={"db": FROZEN_CONFIG.database_sha256},
        population_manifest_sha256=FROZEN_CONFIG.population_manifest_sha256,
        frozen_config_sha256="1" * 64,
        perturbation_contract_sha256="2" * 64,
        acceleration_contract_sha256="3" * 64,
    )


def _target_provenance() -> Exp414CheckpointProvenance:
    return Exp414CheckpointProvenance(
        git_commit="a" * 40,
        database_snapshot={"db": FROZEN_CONFIG.database_sha256},
        population_manifest_sha256=FROZEN_CONFIG.population_manifest_sha256,
        frozen_config_sha256="1" * 64,
        perturbation_contract_sha256="2" * 64,
        acceleration_contract_sha256="3" * 64,
        missing_repair_contract_sha256="4" * 64,
        missing_inventory_sha256="5" * 64,
        qualification_contract_sha256="6" * 64,
        exp411_migration_source_sha256="7" * 64,
    )


def _build_source(
    parent: Path,
    prepared_by_dataset: dict[str, PreparedDataset],
    *,
    provenance: CheckpointProvenance | None = None,
    qualification_overrides: dict[str, DatasetQualification] | None = None,
    identity_overrides: dict[str, str] | None = None,
) -> tuple[Path, tuple[str, ...]]:
    datasets = tuple(sorted(prepared_by_dataset))
    root = parent / "exp411"
    store = CheckpointStore(
        root,
        provenance or _source_provenance(),
        datasets,
    )
    for dataset in datasets:
        prepared = prepared_by_dataset[dataset]
        store.write_dataset(
            (qualification_overrides or {}).get(dataset, _qualification(prepared)),
            _backend_audit(),
            (identity_overrides or {}).get(
                dataset,
                dataset_train_identity_sha256(prepared.runs),
            ),
            0.5,
        )
    return root, datasets


def _target(parent: Path, datasets: tuple[str, ...]) -> Exp414CheckpointStore:
    return Exp414CheckpointStore(
        parent / "exp414",
        _target_provenance(),
        datasets,
    )


class Exp411MigrationTests(unittest.TestCase):
    def test_clean_matching_identity_migrates_as_verified(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, datasets)

            migrated = migrate_if_eligible(
                prepared[TARGET],
                source,
                target,
                evidence,
            )

            self.assertTrue(migrated)
            record = target.load_dataset_record(TARGET)
            self.assertEqual(
                record["execution_source"],
                "migrated_exp411_verified",
            )
            self.assertEqual(record["qualification"], _qualification(prepared[TARGET]))
            expected_audit_hash = sha256(
                canonical_json_bytes(asdict(prepared[TARGET].repair_audit))
            ).hexdigest()
            self.assertEqual(record["repair_audit_sha256"], expected_audit_hash)

    def test_dataset_with_any_repair_is_never_migrated(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        repaired = _prepared(TARGET, repaired=True)
        prepared[TARGET] = repaired
        identity = repaired.repair_audit.repaired_train_identity_sha256
        assert identity is not None
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(
                parent,
                prepared,
                identity_overrides={TARGET: identity},
            )
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, datasets)

            self.assertFalse(
                migrate_if_eligible(repaired, source, target, evidence)
            )
            self.assertFalse(target.has_dataset(TARGET))

    def test_raw_identity_mismatch_returns_false_without_writing(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(
                parent,
                prepared,
                identity_overrides={TARGET: "f" * 64},
            )
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, datasets)

            self.assertFalse(
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    evidence,
                )
            )
            self.assertFalse(target.has_dataset(TARGET))

    def test_absent_source_record_returns_false_after_validating_source(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        source_prepared = {
            dataset: prepared[dataset]
            for dataset in REPRESENTATIVES
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, source_datasets = _build_source(parent, source_prepared)
            source = open_exp411_checkpoint(root, source_datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, tuple(sorted(prepared)))

            self.assertFalse(
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    evidence,
                )
            )
            self.assertFalse(target.has_dataset(TARGET))

    def test_invalid_exp411_root_provenance_fails_source_open(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(
                parent,
                prepared,
                provenance=_source_provenance(git_commit="f" * 40),
            )

            with self.assertRaisesRegex(ValueError, "Git commit"):
                open_exp411_checkpoint(root, datasets)

    def test_stale_progress_is_ignored_and_valid_records_are_counted(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            (root / "progress.json").write_text(
                json.dumps(
                    {
                        "completed_datasets": 3,
                        "current_dataset": TARGET,
                        "expected_datasets": 4,
                    }
                ),
                encoding="utf-8",
            )

            source = open_exp411_checkpoint(root, datasets)

            self.assertEqual(len(source.load_completed()), 4)
            self.assertIn(TARGET, source.load_completed())

    def test_representatives_are_fixed_and_return_canonical_payload_hashes(self) -> None:
        self.assertEqual(EXP411_MIGRATION_REPRESENTATIVES, REPRESENTATIVES)
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)

            evidence = verify_migration_representatives(source, prepared)

            self.assertEqual(tuple(evidence), REPRESENTATIVES)
            self.assertEqual(
                evidence,
                {
                    dataset: source.load_dataset_record(dataset)["payload_sha256"]
                    for dataset in REPRESENTATIVES
                },
            )

    def test_representative_qualification_mismatch_blocks_all_migration(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        mismatch = replace(
            _qualification(prepared[REPRESENTATIVES[0]]),
            identity_group_count=4,
        )
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(
                parent,
                prepared,
                qualification_overrides={REPRESENTATIVES[0]: mismatch},
            )
            source = open_exp411_checkpoint(root, datasets)
            target = _target(parent, datasets)

            with self.assertRaisesRegex(ValueError, "representative qualification"):
                verify_migration_representatives(source, prepared)
            with self.assertRaisesRegex(ValueError, "representative evidence"):
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    {},
                )
            self.assertFalse(target.has_dataset(TARGET))

    def test_prepared_run_identity_mismatch_blocks_representative_gate(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)
            original = prepared[REPRESENTATIVES[0]]
            forged_runs = _prepared(
                REPRESENTATIVES[0],
                offset=100.0,
            ).runs
            prepared[REPRESENTATIVES[0]] = PreparedDataset(
                dataset=original.dataset,
                runs=forged_runs,
                repair_audit=original.repair_audit,
            )

            with self.assertRaisesRegex(ValueError, "not clean"):
                verify_migration_representatives(source, prepared)

    def test_source_record_corruption_raises_instead_of_recomputing(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, datasets)
            path = root / "datasets" / (
                sha256(TARGET.encode("utf-8")).hexdigest() + ".json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["payload_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "payload hash"):
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    evidence,
                )
            self.assertFalse(target.has_dataset(TARGET))


if __name__ == "__main__":
    unittest.main()
