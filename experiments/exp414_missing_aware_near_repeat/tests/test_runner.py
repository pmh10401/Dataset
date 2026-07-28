from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    qualify_dataset,
)
from experiments.exp411_near_repeat_qualification.perturbations import (
    perturbation_types,
)
from experiments.exp414_missing_aware_near_repeat.checkpoints import (
    Exp414CheckpointProvenance,
    Exp414CheckpointStore,
)
from experiments.exp414_missing_aware_near_repeat.contracts import (
    MissingInventory,
    PreparedDataset,
)
from experiments.exp414_missing_aware_near_repeat.config import FROZEN_CONFIG
from experiments.exp414_missing_aware_near_repeat.repair import (
    build_dataset_repair_audit,
    repair_train_blob,
)
from experiments.exp414_missing_aware_near_repeat.runner import (
    _build_smoke_prepared,
    _run_checkpointed_stream,
    aggregate_subgroups,
    missing_support_abstention,
    qualify_prepared_dataset,
    run_full_qualification,
    run_representative_smoke,
)


def _prepared(
    dataset: str,
    payloads: tuple[tuple[float, ...], ...],
) -> PreparedDataset:
    runs = []
    audits = []
    for index, values in enumerate(payloads):
        run, audit = repair_train_blob(
            dataset,
            index,
            np.asarray(values, dtype="<f4").tobytes(),
        )
        if run is not None:
            runs.append(run)
        audits.append(audit)
    repair_audit = build_dataset_repair_audit(dataset, runs, audits)
    return PreparedDataset(
        dataset=dataset,
        runs=tuple(runs) if repair_audit.status == "ready" else (),
        repair_audit=repair_audit,
    )


def _clean(dataset: str, offset: float = 0.0) -> PreparedDataset:
    return _prepared(
        dataset,
        tuple(
            tuple(offset + index + value for value in range(8))
            for index in range(6)
        ),
    )


def _target_provenance() -> Exp414CheckpointProvenance:
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


def _repair_digest(prepared: PreparedDataset) -> str:
    return sha256(
        canonical_json_bytes(asdict(prepared.repair_audit))
    ).hexdigest()


class QualificationTests(unittest.TestCase):
    def test_all_missing_run_abstains_entire_dataset(self) -> None:
        prepared = _prepared(
            "AllMissing_normal_0",
            (
                (0.0, 1.0, 2.0),
                (1.0, 2.0, 3.0),
                (2.0, 3.0, 4.0),
                (3.0, 4.0, 5.0),
                (4.0, 5.0, 6.0),
                (np.nan, np.inf, -np.inf),
            ),
        )

        row = missing_support_abstention(prepared)

        self.assertEqual(row.normal_total, 6)
        self.assertEqual(row.normal_abstained, 6)
        self.assertEqual(row.identity_group_count, 6)
        self.assertTrue(all(value == 0 for value in row.normal_near.values()))
        self.assertTrue(all(value == 0 for value in row.synthetic_total.values()))
        self.assertEqual(
            tuple(row.normal_near),
            ("0.01", "0.05", "0.10"),
        )
        self.assertEqual(
            set(row.perturbation_total["0.01"]),
            set(perturbation_types()),
        )

    def test_clean_prepared_dataset_matches_exp411(self) -> None:
        prepared = _clean("Clean_normal_0")
        expected = qualify_dataset(
            dataset=prepared.dataset,
            family="Clean",
            runs=prepared.runs,
        )

        observed = qualify_prepared_dataset(prepared, accelerated=False)

        self.assertEqual(observed, expected)

    def test_prepared_values_must_match_the_bound_repaired_identity(self) -> None:
        prepared = _clean("Clean_normal_0")
        changed = replace(
            prepared,
            runs=(
                replace(
                    prepared.runs[0],
                    values=tuple(
                        value + 100.0
                        for value in prepared.runs[0].values
                    ),
                ),
                *prepared.runs[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "identity"):
            qualify_prepared_dataset(changed, accelerated=False)

    def test_subgroups_are_diagnostic_and_only_all_has_selection_authority(
        self,
    ) -> None:
        clean = _clean("Alpha_normal_0")
        repaired = _prepared(
            "DodgerLoopDay_normal_0",
            tuple(
                (float(index), np.nan, float(index + 2), float(index + 3))
                if index == 0
                else tuple(float(index + value) for value in range(4))
                for index in range(6)
            ),
        )
        abstained = _prepared(
            "MelbournePedestrian_normal_0",
            (
                (0.0, 1.0, 2.0),
                (1.0, 2.0, 3.0),
                (2.0, 3.0, 4.0),
                (3.0, 4.0, 5.0),
                (4.0, 5.0, 6.0),
                (np.nan, np.nan, np.nan),
            ),
        )
        rows = (
            qualify_prepared_dataset(clean, accelerated=False),
            qualify_prepared_dataset(repaired, accelerated=False),
            missing_support_abstention(abstained),
        )

        observed = aggregate_subgroups(
            rows,
            (
                clean.repair_audit,
                repaired.repair_audit,
                abstained.repair_audit,
            ),
        )

        self.assertEqual(
            tuple(observed),
            (
                "all",
                "nonmissing",
                "missing_repaired",
                "DodgerLoop",
                "MelbournePedestrian",
                "missing_support_abstain",
            ),
        )
        self.assertTrue(observed["all"]["selection_authority"])
        self.assertEqual(observed["all"]["dataset_count"], 3)
        for name in tuple(observed)[1:]:
            self.assertFalse(observed[name]["selection_authority"])
            self.assertEqual(observed[name]["dataset_count"], 1)


class StreamExecutionTests(unittest.TestCase):
    def test_stream_migrates_repairs_abstains_and_completes_from_records(
        self,
    ) -> None:
        migrated = _clean("Alpha_normal_0")
        repaired = _prepared(
            "Beta_normal_0",
            tuple(
                (float(index), np.nan, float(index + 2), float(index + 3))
                if index == 0
                else tuple(float(index + value) for value in range(4))
                for index in range(6)
            ),
        )
        abstained = _prepared(
            "Gamma_normal_0",
            (
                (0.0, 1.0, 2.0),
                (1.0, 2.0, 3.0),
                (2.0, 3.0, 4.0),
                (3.0, 4.0, 5.0),
                (4.0, 5.0, 6.0),
                (np.nan, np.nan, np.nan),
            ),
        )
        prepared = (migrated, repaired, abstained)
        datasets = tuple(row.dataset for row in prepared)
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _target_provenance(),
                datasets,
            )
            store.write_progress(current_dataset="Gamma_normal_0")

            def migrate(
                row: PreparedDataset,
                source: object,
                target: Exp414CheckpointStore,
                evidence: object,
            ) -> bool:
                if row.dataset != "Alpha_normal_0":
                    return False
                result = qualify_prepared_dataset(row, accelerated=False)
                target.write_dataset(
                    result,
                    DatasetBackendAudit(
                        backend="numpy_cpu",
                        sampled_pairs=(),
                        max_abs_error=0.0,
                        fallback_reason=None,
                    ),
                    raw_train_identity_sha256=(
                        row.repair_audit.raw_train_identity_sha256
                    ),
                    repaired_train_identity_sha256=(
                        row.repair_audit.repaired_train_identity_sha256
                    ),
                    repair_audit_sha256=_repair_digest(row),
                    execution_source="migrated_exp411_verified",
                    elapsed_seconds=0.1,
                )
                return True

            with patch(
                "experiments.exp414_missing_aware_near_repeat.runner."
                "migrate_if_eligible",
                side_effect=migrate,
            ):
                rows, audits, subgroups = _run_checkpointed_stream(
                    prepared,
                    store,
                    source_store=object(),
                    representative_evidence={},
                    migration_enabled=True,
                    expected_families=3,
                )

            self.assertEqual(len(rows), 3)
            self.assertEqual(len(audits), 3)
            self.assertEqual(subgroups["all"]["dataset_count"], 3)
            self.assertEqual(store.progress()["completed_datasets"], 3)
            self.assertEqual(
                store.load_dataset_record("Alpha_normal_0")["execution_source"],
                "migrated_exp411_verified",
            )
            self.assertEqual(
                store.load_dataset_record("Beta_normal_0")["execution_source"],
                "exp414_computed",
            )
            gamma = store.load_dataset_record("Gamma_normal_0")
            self.assertEqual(
                gamma["backend_audit"].fallback_reason,
                "missing_support_abstain",
            )

    def test_existing_record_identity_mismatch_fails_closed(self) -> None:
        prepared = _clean("Alpha_normal_0")
        with TemporaryDirectory() as raw:
            store = Exp414CheckpointStore(
                Path(raw) / "checkpoint",
                _target_provenance(),
                (prepared.dataset,),
            )
            row = qualify_prepared_dataset(prepared, accelerated=False)
            store.write_dataset(
                row,
                DatasetBackendAudit(
                    backend="numpy_cpu",
                    sampled_pairs=(),
                    max_abs_error=0.0,
                    fallback_reason=None,
                ),
                raw_train_identity_sha256="f" * 64,
                repaired_train_identity_sha256=(
                    prepared.repair_audit.repaired_train_identity_sha256
                ),
                repair_audit_sha256=_repair_digest(prepared),
                execution_source="exp414_computed",
                elapsed_seconds=0.1,
            )

            with self.assertRaisesRegex(ValueError, "TRAIN identity mismatch"):
                _run_checkpointed_stream(
                    (prepared,),
                    store,
                    migration_enabled=False,
                    expected_families=1,
                )


class EntryPointTests(unittest.TestCase):
    def test_smoke_covers_every_missingness_regime_before_finalization(
        self,
    ) -> None:
        prepared = _build_smoke_prepared()
        observed_methods = {
            audit.repair_method
            for row in prepared
            for audit in row.repair_audit.runs
        }
        repaired_hashes = [
            audit.repaired_values_sha256
            for row in prepared
            if row.dataset.startswith("SmokeCollision_")
            for audit in row.repair_audit.runs
        ]

        self.assertEqual(len(prepared), 5)
        self.assertIn("unchanged", observed_methods)
        self.assertIn("linear_and_edge_nearest", observed_methods)
        self.assertIn("all_missing_abstain", observed_methods)
        self.assertLess(len(set(repaired_hashes)), len(repaired_hashes))

        with TemporaryDirectory() as raw:
            output = Path(raw) / "publication"
            captured: dict[str, object] = {}

            def finalize(**values: object) -> dict[str, str]:
                captured.update(values)
                return {"manifest": "f" * 64}

            with patch(
                "experiments.exp414_missing_aware_near_repeat.runner."
                "_finalize_outputs",
                side_effect=finalize,
            ):
                hashes = run_representative_smoke(output)

            self.assertEqual(hashes, {"manifest": "f" * 64})
            self.assertEqual(
                captured["provenance"]["smoke_coverage"],
                (
                    "clean",
                    "internal_gap",
                    "edge_gap",
                    "repaired_identity_collision",
                    "all_missing",
                ),
            )
            self.assertFalse(output.exists())

    def test_smoke_rejects_candidate_trace_mutation_with_equal_counts(
        self,
    ) -> None:
        real_qualify = qualify_dataset

        def mutate_accelerated_trace(**values: object):
            row = real_qualify(**values)
            trace = values.get("parity_trace")
            if values.get("accelerated") and trace:
                trace[-1] = {
                    **trace[-1],
                    "candidate_index": int(trace[-1]["candidate_index"]) + 1,
                    "tag": (
                        "near"
                        if trace[-1]["tag"] != "near"
                        else "unfamiliar"
                    ),
                }
            return row

        with TemporaryDirectory() as raw:
            output = Path(raw) / "publication"
            with (
                patch(
                    "experiments.exp414_missing_aware_near_repeat.runner."
                    "qualify_dataset",
                    side_effect=mutate_accelerated_trace,
                ),
                patch(
                    "experiments.exp414_missing_aware_near_repeat.runner."
                    "_finalize_outputs",
                    return_value={"manifest": "f" * 64},
                ),
            ):
                with self.assertRaisesRegex(ValueError, "tag parity"):
                    run_representative_smoke(output)
            self.assertFalse(output.exists())

    def test_full_preflight_git_failure_never_creates_output(self) -> None:
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"
            with patch(
                "experiments.exp414_missing_aware_near_repeat.runner."
                "_require_clean_git",
                side_effect=ValueError("dirty"),
            ):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    run_full_qualification(
                        parent / "db.sqlite",
                        parent / "population.json",
                        output,
                        exp411_checkpoint_dir=parent / "exp411",
                        checkpoint_dir=parent / "exp414",
                        benchmark_report=parent / "benchmark.json",
                    )
            self.assertFalse(output.exists())

    def test_full_inventory_failure_never_creates_output(self) -> None:
        invalid = replace(
            self._valid_inventory(),
            affected_runs=FROZEN_CONFIG.expected_missing_runs + 1,
        )
        self._assert_full_failure_leaves_no_output(
            "missing inventory",
            inventory=invalid,
        )

    def test_full_benchmark_failure_never_creates_output(self) -> None:
        self._assert_full_failure_leaves_no_output(
            "benchmark failure",
            benchmark_error=ValueError("benchmark failure"),
        )

    def test_full_stream_failure_never_creates_output(self) -> None:
        self._assert_full_failure_leaves_no_output(
            "stream failure",
            stream_error=ValueError("stream failure"),
        )

    def test_full_database_recheck_failure_never_creates_output(self) -> None:
        self._assert_full_failure_leaves_no_output(
            "snapshot changed",
            database_recheck_failure=True,
        )

    def _assert_full_failure_leaves_no_output(
        self,
        message: str,
        *,
        inventory: MissingInventory | None = None,
        benchmark_error: ValueError | None = None,
        stream_error: ValueError | None = None,
        database_recheck_failure: bool = False,
    ) -> None:
        pinned = {"db": FROZEN_CONFIG.database_sha256}
        snapshots = (
            [pinned, {"db": "f" * 64}]
            if database_recheck_failure
            else [pinned]
        )
        contracts = {
            "frozen_config_sha256": "1" * 64,
            "perturbation_contract_sha256": "2" * 64,
            "acceleration_contract_sha256": "3" * 64,
            "missing_repair_contract_sha256": "4" * 64,
            "qualification_contract_sha256": "5" * 64,
            "execution_contract_sha256": "6" * 64,
        }
        with TemporaryDirectory() as raw:
            parent = Path(raw)
            output = parent / "publication"
            with ExitStack() as stack:
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_require_clean_git",
                        return_value="a" * 40,
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "configure_runtime",
                        return_value={
                            "numba_threads": 12,
                            "nested_threads": 1,
                        },
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_load_population_manifest",
                        return_value=("Alpha_normal_0",),
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "sha256_file",
                        return_value="b" * 64,
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "database_snapshot",
                        side_effect=snapshots,
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "scan_missing_inventory",
                        return_value=inventory or self._valid_inventory(),
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_contract_values",
                        return_value=contracts,
                    )
                )
                benchmark = stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_load_benchmark_report",
                        return_value={"status": "passed"},
                    )
                )
                if benchmark_error is not None:
                    benchmark.side_effect = benchmark_error
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "open_exp411_checkpoint",
                        return_value=object(),
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "iter_prepared_train_datasets",
                        return_value=(),
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "verify_migration_representatives",
                        return_value={},
                    )
                )
                stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_migration_source_sha256",
                        return_value="c" * 64,
                    )
                )
                stream = stack.enter_context(
                    patch(
                        "experiments.exp414_missing_aware_near_repeat.runner."
                        "_run_checkpointed_stream",
                        return_value=((), (), {"all": {}}),
                    )
                )
                if stream_error is not None:
                    stream.side_effect = stream_error

                with self.assertRaisesRegex(ValueError, message):
                    run_full_qualification(
                        parent / "db.sqlite",
                        parent / "population.json",
                        output,
                        exp411_checkpoint_dir=parent / "exp411",
                        checkpoint_dir=parent / "exp414",
                        benchmark_report=parent / "benchmark.json",
                    )
            self.assertFalse(output.exists())

    @staticmethod
    def _valid_inventory() -> MissingInventory:
        return MissingInventory(
            datasets=FROZEN_CONFIG.expected_datasets,
            affected_datasets=FROZEN_CONFIG.expected_missing_datasets,
            affected_runs=FROZEN_CONFIG.expected_missing_runs,
            nonfinite_values=FROZEN_CONFIG.expected_nonfinite_values,
            all_missing_runs=FROZEN_CONFIG.expected_all_missing_runs,
            affected_datasets_sha256=(
                FROZEN_CONFIG.affected_datasets_sha256
            ),
        )


if __name__ == "__main__":
    unittest.main()
