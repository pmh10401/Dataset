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
    TrainRun,
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
EXP411_FROZEN_CONFIG_SHA256 = (
    "e4deeaf657e63dc1280e09c88a933cdb8ae0d103a25fe98b7bd894df50b1e588"
)
EXP411_PERTURBATION_CONTRACT_SHA256 = (
    "6bc750fff507189ec190a04749be2f99f4fb766cba76d70804bf7cc713f1aeff"
)
EXP411_ACCELERATION_CONTRACT_SHA256 = (
    "4e5a793684b576b465bfcf2606b58d99a67671a7c4fbc6a348bf4bb9f52e9d79"
)


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
    frozen_config_sha256: str = EXP411_FROZEN_CONFIG_SHA256,
    perturbation_contract_sha256: str = (
        EXP411_PERTURBATION_CONTRACT_SHA256
    ),
    acceleration_contract_sha256: str = EXP411_ACCELERATION_CONTRACT_SHA256,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        git_commit=git_commit,
        database_snapshot={"db": FROZEN_CONFIG.database_sha256},
        population_manifest_sha256=FROZEN_CONFIG.population_manifest_sha256,
        frozen_config_sha256=frozen_config_sha256,
        perturbation_contract_sha256=perturbation_contract_sha256,
        acceleration_contract_sha256=acceleration_contract_sha256,
    )


def _target_provenance(
    migration_source_sha256: str = "7" * 64,
) -> Exp414CheckpointProvenance:
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
        exp411_migration_source_sha256=migration_source_sha256,
    )


def _build_source(
    parent: Path,
    prepared_by_dataset: dict[str, PreparedDataset],
    *,
    provenance: CheckpointProvenance | None = None,
    qualification_overrides: dict[str, DatasetQualification] | None = None,
    identity_overrides: dict[str, str] | None = None,
    backend_audit_overrides: dict[str, DatasetBackendAudit] | None = None,
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
            (backend_audit_overrides or {}).get(dataset, _backend_audit()),
            (identity_overrides or {}).get(
                dataset,
                dataset_train_identity_sha256(prepared.runs),
            ),
            0.5,
        )
    return root, datasets


def _migration_source_digest(
    source: CheckpointStore,
    evidence: dict[str, str] | object,
) -> str:
    return sha256(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "source_checkpoint_provenance": source.provenance.to_payload(),
                "source_expected_datasets": list(source.expected_datasets),
                "representative_payload_sha256": dict(evidence),
            }
        )
    ).hexdigest()


def _target(
    parent: Path,
    datasets: tuple[str, ...],
    source: CheckpointStore,
    evidence: object,
    *,
    migration_source_sha256: str | None = None,
) -> Exp414CheckpointStore:
    source_digest = (
        _migration_source_digest(source, evidence)
        if migration_source_sha256 is None
        else migration_source_sha256
    )
    return Exp414CheckpointStore(
        parent / "exp414",
        _target_provenance(source_digest),
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
            target = _target(parent, datasets, source, evidence)

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
            target = _target(parent, datasets, source, evidence)

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
            target = _target(parent, datasets, source, evidence)

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
            target = _target(
                parent,
                tuple(sorted(prepared)),
                source,
                evidence,
            )

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

    def test_each_exp411_contract_digest_mutation_fails_source_open(
        self,
    ) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        mutations = (
            ("frozen_config_sha256", "frozen config"),
            ("perturbation_contract_sha256", "perturbation contract"),
            ("acceleration_contract_sha256", "acceleration contract"),
        )
        for field, message in mutations:
            with self.subTest(field=field), TemporaryDirectory() as raw:
                parent = Path(raw)
                provenance = replace(
                    _source_provenance(),
                    **{field: "f" * 64},
                )
                root, datasets = _build_source(
                    parent,
                    prepared,
                    provenance=provenance,
                )

                with self.assertRaisesRegex(ValueError, message):
                    open_exp411_checkpoint(root, datasets)

    def test_each_exp411_contract_digest_is_checked_again_on_revalidation(
        self,
    ) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        mutations = (
            ("frozen_config_sha256", "frozen config"),
            ("perturbation_contract_sha256", "perturbation contract"),
            ("acceleration_contract_sha256", "acceleration contract"),
        )
        for field, message in mutations:
            with self.subTest(field=field), TemporaryDirectory() as raw:
                parent = Path(raw)
                root, datasets = _build_source(parent, prepared)
                source = open_exp411_checkpoint(root, datasets)
                source.provenance = replace(
                    source.provenance,
                    **{field: "f" * 64},
                )

                with self.assertRaisesRegex(ValueError, message):
                    verify_migration_representatives(source, prepared)

    def test_source_backend_audit_must_pass_ready_record_gate(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in REPRESENTATIVES
        }
        invalid_audits = (
            (
                replace(
                    _backend_audit(),
                    max_abs_error=(
                        EXP411_FROZEN_CONFIG.distance_tolerance * 1.000001
                    ),
                ),
                "tolerance",
            ),
            (
                replace(
                    _backend_audit(),
                    fallback_reason="pair_audit_tolerance_exceeded",
                ),
                "fallback",
            ),
        )
        for audit, message in invalid_audits:
            with self.subTest(message=message), TemporaryDirectory() as raw:
                parent = Path(raw)
                root, datasets = _build_source(
                    parent,
                    prepared,
                    backend_audit_overrides={
                        REPRESENTATIVES[0]: audit,
                    },
                )

                with self.assertRaisesRegex(ValueError, message):
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

            with self.assertRaisesRegex(ValueError, "representative qualification"):
                verify_migration_representatives(source, prepared)
            forged_evidence = {
                dataset: source.load_dataset_record(dataset)["payload_sha256"]
                for dataset in REPRESENTATIVES
            }
            target = _target(
                parent,
                datasets,
                source,
                forged_evidence,
            )
            with self.assertRaisesRegex(ValueError, "representative evidence"):
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    forged_evidence,
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

    def test_changed_values_with_old_hash_are_not_migrated(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(parent, datasets, source, evidence)
            original = prepared[TARGET]
            first = original.runs[0]
            forged_run = TrainRun(
                dataset=first.dataset,
                instance_index=first.instance_index,
                values=tuple(value + 100.0 for value in first.values),
                values_sha256=first.values_sha256,
            )
            forged = PreparedDataset(
                dataset=original.dataset,
                runs=(forged_run, *original.runs[1:]),
                repair_audit=original.repair_audit,
            )

            self.assertFalse(
                migrate_if_eligible(
                    forged,
                    source,
                    target,
                    evidence,
                )
            )
            self.assertFalse(target.has_dataset(TARGET))

    def test_target_migration_source_digest_must_match_verified_evidence(self) -> None:
        prepared = {
            dataset: _prepared(dataset)
            for dataset in (*REPRESENTATIVES, TARGET)
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            root, datasets = _build_source(parent, prepared)
            source = open_exp411_checkpoint(root, datasets)
            evidence = verify_migration_representatives(source, prepared)
            target = _target(
                parent,
                datasets,
                source,
                evidence,
                migration_source_sha256="f" * 64,
            )

            with self.assertRaisesRegex(ValueError, "source binding"):
                migrate_if_eligible(
                    prepared[TARGET],
                    source,
                    target,
                    evidence,
                )
            self.assertFalse(target.has_dataset(TARGET))

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
            target = _target(parent, datasets, source, evidence)
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
