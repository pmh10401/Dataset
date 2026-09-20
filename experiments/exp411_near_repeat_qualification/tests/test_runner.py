from dataclasses import replace
import csv
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp411_near_repeat_qualification.checkpoints import (
    CheckpointProvenance,
    CheckpointStore,
)
from experiments.exp411_near_repeat_qualification.acceleration import (
    acceleration_contract_sha256,
)
from experiments.exp411_near_repeat_qualification.config import FROZEN_CONFIG
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    TrainRun,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    qualify_dataset,
)
from experiments.exp411_near_repeat_qualification.runner import (
    _require_immutable_file,
    _require_runtime_contract,
    _run_checkpointed_stream,
    dataset_train_identity_sha256,
    load_benchmark_report,
    load_population_manifest,
    run_representative_smoke,
)


class RunnerTests(unittest.TestCase):
    def test_prepublication_file_hash_rejects_mutation(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "input.json"
            path.write_bytes(b"original")
            expected = sha256(path.read_bytes()).hexdigest()
            _require_immutable_file(path, expected, "input")
            path.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "input"):
                _require_immutable_file(path, expected, "input")

    def test_full_runtime_contract_rejects_wrong_thread_state(self) -> None:
        _require_runtime_contract({
            "numba_threads": 12,
            "nested_threads": 1,
        })
        for invalid in (
            {"numba_threads": 11, "nested_threads": 1},
            {"numba_threads": 12, "nested_threads": 2},
        ):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError,
                "runtime",
            ):
                _require_runtime_contract(invalid)

    @staticmethod
    def _runs(dataset: str, offset: int = 0) -> tuple[TrainRun, ...]:
        return tuple(
            TrainRun(
                dataset=dataset,
                instance_index=index,
                values=(
                    0.0,
                    (index + offset) * 0.1,
                    (index + offset) ** 2 * 0.02,
                    (index + offset) * 0.3,
                ),
                values_sha256=f"{index + offset:064x}",
            )
            for index in range(6)
        )

    @staticmethod
    def _checkpoint_provenance() -> CheckpointProvenance:
        return CheckpointProvenance(
            git_commit="1" * 40,
            database_snapshot={"db": "2" * 64},
            population_manifest_sha256="3" * 64,
            frozen_config_sha256="4" * 64,
            perturbation_contract_sha256="5" * 64,
            acceleration_contract_sha256="6" * 64,
        )

    def test_checkpointed_stream_skips_matching_completed_dataset(self) -> None:
        alpha_runs = self._runs("Alpha_normal_0")
        beta_runs = self._runs("Beta_normal_0", offset=10)
        with TemporaryDirectory() as raw:
            store = CheckpointStore(
                Path(raw) / "checkpoint",
                self._checkpoint_provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )
            alpha_audits: list[DatasetBackendAudit] = []
            alpha = qualify_dataset(
                dataset="Alpha_normal_0",
                family="Alpha",
                runs=alpha_runs,
                accelerated=True,
                backend_audit=alpha_audits,
            )
            store.write_dataset(
                alpha,
                alpha_audits[0],
                dataset_train_identity_sha256(alpha_runs),
                1.0,
            )

            rows = _run_checkpointed_stream(
                (
                    ("Alpha_normal_0", alpha_runs),
                    ("Beta_normal_0", beta_runs),
                ),
                store,
                config=FROZEN_CONFIG,
            )

            self.assertEqual([row.dataset for row in rows], [
                "Alpha_normal_0",
                "Beta_normal_0",
            ])
            self.assertEqual(
                set(store.load_completed()),
                {"Alpha_normal_0", "Beta_normal_0"},
            )

    def test_checkpointed_stream_rejects_train_identity_mismatch(self) -> None:
        runs = self._runs("Alpha_normal_0")
        with TemporaryDirectory() as raw:
            store = CheckpointStore(
                Path(raw) / "checkpoint",
                self._checkpoint_provenance(),
                ("Alpha_normal_0",),
            )
            audits: list[DatasetBackendAudit] = []
            row = qualify_dataset(
                dataset="Alpha_normal_0",
                family="Alpha",
                runs=runs,
                accelerated=True,
                backend_audit=audits,
            )
            store.write_dataset(row, audits[0], "f" * 64, 1.0)

            with self.assertRaisesRegex(ValueError, "identity"):
                _run_checkpointed_stream(
                    (("Alpha_normal_0", runs),),
                    store,
                    config=FROZEN_CONFIG,
                )

    def test_stream_error_preserves_previous_checkpoint(self) -> None:
        alpha_runs = self._runs("Alpha_normal_0")
        invalid_beta_runs = self._runs("Foreign_normal_0", offset=10)
        with TemporaryDirectory() as raw:
            store = CheckpointStore(
                Path(raw) / "checkpoint",
                self._checkpoint_provenance(),
                ("Alpha_normal_0", "Beta_normal_0"),
            )

            with self.assertRaisesRegex(ValueError, "foreign"):
                _run_checkpointed_stream(
                    (
                        ("Alpha_normal_0", alpha_runs),
                        ("Beta_normal_0", invalid_beta_runs),
                    ),
                    store,
                    config=FROZEN_CONFIG,
                )

            self.assertEqual(
                set(store.load_completed()),
                {"Alpha_normal_0"},
            )

    def test_train_identity_hash_is_ordered_and_content_sensitive(self) -> None:
        first = TrainRun("Alpha_normal_0", 0, (0.0, 1.0), "1" * 64)
        second = TrainRun("Alpha_normal_0", 1, (2.0, 3.0), "2" * 64)

        original = dataset_train_identity_sha256((first, second))
        repeated = dataset_train_identity_sha256((first, second))
        reversed_order = dataset_train_identity_sha256((second, first))

        self.assertEqual(original, repeated)
        self.assertNotEqual(original, reversed_order)

    def test_benchmark_report_loader_rejects_failed_adoption(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_datasets": {},
                        "runtime": {
                            "numba_threads": 12,
                            "nested_threads": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset"):
                load_benchmark_report(path)

    def test_benchmark_report_loader_rejects_foreign_acceleration_code(self) -> None:
        row = {
            "numpy_seconds": 4.0,
            "numba_cold_seconds": 2.0,
            "numba_warm_seconds": 1.5,
            "warm_speedup": 2.6666666666666665,
            "max_abs_error": 1e-12,
            "tag_identity": True,
            "selected_quantile_identity": True,
        }
        with TemporaryDirectory() as raw:
            path = Path(raw) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_datasets": {
                            "small": {
                                "dataset": "ACSF1_normal_0",
                                **row,
                            },
                            "medium": {
                                "dataset": "HandOutlines_normal_0",
                                **row,
                            },
                            "large": {
                                "dataset": "StarLightCurves_normal_3",
                                **row,
                            },
                        },
                        "runtime": {
                            "numba_threads": 12,
                            "nested_threads": 1,
                        },
                        "acceleration_contract_sha256": "f" * 64,
                    }
                ),
                encoding="utf-8",
            )

            self.assertNotEqual(
                "f" * 64,
                acceleration_contract_sha256(),
            )
            with self.assertRaisesRegex(ValueError, "contract"):
                load_benchmark_report(path)

    def test_benchmark_report_loader_rejects_foreign_git_commit(self) -> None:
        row = {
            "numpy_seconds": 4.0,
            "numba_cold_seconds": 2.0,
            "numba_warm_seconds": 1.5,
            "warm_speedup": 2.6666666666666665,
            "max_abs_error": 1e-12,
            "tag_identity": True,
            "selected_quantile_identity": True,
        }
        with TemporaryDirectory() as raw:
            path = Path(raw) / "benchmark.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "benchmark_datasets": {
                            "small": {
                                "dataset": "ACSF1_normal_0",
                                **row,
                            },
                            "medium": {
                                "dataset": "HandOutlines_normal_0",
                                **row,
                            },
                            "large": {
                                "dataset": "StarLightCurves_normal_3",
                                **row,
                            },
                        },
                        "runtime": {
                            "numba_threads": 12,
                            "nested_threads": 1,
                        },
                        "acceleration_contract_sha256": (
                            acceleration_contract_sha256()
                        ),
                        "git_commit": "f" * 40,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "commit"):
                load_benchmark_report(path, expected_commit="1" * 40)

    def test_representative_smoke_publishes_complete_manifest(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "smoke"
            hashes = run_representative_smoke(output)
            manifest = json.loads(
                (output / "MANIFEST.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (output / "exp411_summary.json").read_text(encoding="utf-8")
            )
            backend = json.loads(
                (output / "exp411_backend_assessment.json").read_text(
                    encoding="utf-8"
                )
            )
            with (output / "exp411_dataset_qualification.csv").open(
                encoding="utf-8",
                newline="",
            ) as handle:
                dataset_rows = tuple(csv.DictReader(handle))
        self.assertFalse(summary["full_qualification_executed"])
        self.assertFalse(summary["operational_policy_changed"])
        self.assertFalse(summary["exp412_prediction_created"])
        self.assertEqual(summary["dataset_count"], 32)
        self.assertEqual(summary["family_count"], 4)
        self.assertIn("perturbation_contract", manifest["provenance"])
        self.assertEqual(
            backend["canonical_backend"],
            "numpy_cpu",
        )
        self.assertIn("assessments", backend)
        self.assertEqual(
            set(backend["assessments"]),
            {"numba_cpu", "torch_cpu", "torch_mps"},
        )
        self.assertEqual(
            backend["baseline_tag_sha256"],
            backend["canonical_tag_sha256"],
        )
        self.assertTrue(backend["accelerated_path_trace_identity"])
        self.assertTrue(backend["accelerated_path_selected_quantile_identity"])
        self.assertIn("query_identity", backend["trace_contract"])
        self.assertIn("numba_cold_stage_seconds", backend["timing"])
        self.assertIn("numba_warm_stage_seconds", backend["timing"])
        for assessment in backend["assessments"].values():
            if "candidate_trace_sha256" in assessment:
                self.assertEqual(
                    assessment["candidate_trace_sha256"],
                    backend["baseline_tag_sha256"],
                )
        self.assertEqual(
            set(summary["provenance"]["smoke_coverage"]["identity_regimes"]),
            {"abstain_lt5", "logo_5_19", "five_fold_20_plus"},
        )
        self.assertEqual(
            {int(row["identity_group_count"]) for row in dataset_rows},
            {4, 6, 20},
        )
        self.assertEqual(set(hashes), set(manifest["artifact_sha256"]))
        self.assertEqual(len(hashes), 7)

    def test_existing_output_is_rejected(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "must not exist"):
                run_representative_smoke(output)

    def test_population_manifest_is_hash_verified(self) -> None:
        with TemporaryDirectory() as raw:
            path = Path(raw) / "population.json"
            path.write_text(
                json.dumps(
                    {"frozen_hard_lanes": {"Alpha_normal_0": []}},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            digest = sha256(path.read_bytes()).hexdigest()
            config = replace(
                FROZEN_CONFIG,
                expected_datasets=1,
                expected_families=1,
                population_manifest_sha256=digest,
            )
            self.assertEqual(
                load_population_manifest(path, config=config),
                ("Alpha_normal_0",),
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_population_manifest(
                    path,
                    config=replace(config, population_manifest_sha256="0" * 64),
                )


if __name__ == "__main__":
    unittest.main()
