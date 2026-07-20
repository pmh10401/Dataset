import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp263_264_full_missing_aware_integration import config
from experiments.exp263_264_full_missing_aware_integration.inputs import (
    SourceContractError,
    _indices,
    _required_indices,
    _validate_experiment,
    _validate_manifest,
    _validate_n_test,
    _validate_row_indices,
    _validate_row_schema,
    audit_source_contract,
    load_population,
    main,
    sha256_file,
    select_rows,
)


class ConfigurationContractTest(unittest.TestCase):
    def test_policy_constants_are_preregistered(self) -> None:
        self.assertEqual(config.ALPHA, 0.005)
        self.assertEqual(config.MIN_TRAIN_FOR_PROMOTION, 100)
        self.assertEqual(config.REQUIRED_PROMOTION_AGREEMENT, 3)
        self.assertEqual(config.MASK_VARIANT, "mask_025__seed_intersection")

    def test_source_contract_has_exact_frozen_sources(self) -> None:
        self.assertEqual(
            set(config.SOURCE_CONTRACT),
            {
                "baseline",
                "statistical",
                "structural",
                "change_point",
                "reconstruction",
                "missing_override",
            },
        )
        self.assertEqual(
            {
                name: spec.variant
                for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": "final_lanes",
                "statistical": "combined_40",
                "structural": "structural_combined_440",
                "change_point": "change_point_combined_33",
                "reconstruction": "mask_025__seed_intersection",
                "missing_override": "corrected_four_sources",
            },
        )
        self.assertEqual(
            {
                name: spec.alpha for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": None,
                "statistical": 0.005,
                "structural": 0.005,
                "change_point": 0.005,
                "reconstruction": 0.005,
                "missing_override": 0.005,
            },
        )
        self.assertEqual(
            {
                name: spec.experiment
                for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": "exp253",
                "statistical": "exp252",
                "structural": "exp254",
                "change_point": "exp255",
                "reconstruction": "exp257",
                "missing_override": "exp260_262",
            },
        )
        self.assertEqual(
            {
                name: spec.scope for name, spec in config.SOURCE_CONTRACT.items()
            },
            {
                "baseline": "all_1117",
                "statistical": "all_1117",
                "structural": "all_1117",
                "change_point": "all_1117",
                "reconstruction": "n_train_at_least_100",
                "missing_override": "missing_20",
            },
        )

    def test_database_output_and_source_paths_are_absolute(self) -> None:
        self.assertEqual(
            config.DB_PATH,
            config.Path("/Users/minho/Documents/Dataset/univariate_ts.db"),
        )
        self.assertEqual(
            config.OUTPUT_ROOT,
            config.Path(
                "/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration"
            ),
        )
        self.assertTrue(config.DB_PATH.is_absolute())
        self.assertTrue(config.OUTPUT_ROOT.is_absolute())
        expected_paths = {
            "baseline": "/Users/minho/Documents/Dataset/exp253_guarded_standard_to_hard_expansion/frozen_predictions.jsonl",
            "statistical": "/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/frozen_predictions.jsonl",
            "structural": "/Users/minho/Documents/Dataset/exp254_sktime_structural_features/frozen_predictions.jsonl",
            "change_point": "/Users/minho/Documents/Dataset/exp255_change_point_features/frozen_predictions.jsonl",
            "reconstruction": "/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/frozen_predictions.jsonl",
            "missing_override": "/Users/minho/Documents/Dataset/exp260_262_missing_aware_alert/frozen_predictions.jsonl",
        }
        self.assertEqual(
            {
                name: str(spec.artifact_path)
                for name, spec in config.SOURCE_CONTRACT.items()
            },
            expected_paths,
        )
        for path in (config.DB_PATH, config.OUTPUT_ROOT):
            self.assertTrue(path.is_absolute())
        for spec in config.SOURCE_CONTRACT.values():
            self.assertTrue(spec.artifact_path.is_absolute())


class SourceContractTest(unittest.TestCase):
    def test_source_contract_requires_full_coverage(self) -> None:
        audit = audit_source_contract()

        self.assertEqual(audit["raw_database_dataset_count"], 1119)
        self.assertEqual(
            audit["frozen_excluded_datasets"],
            ["CornellWhaleChallenge", "Wafer_normal_1"],
        )
        self.assertEqual(audit["frozen_population_count"], 1117)
        self.assertEqual(audit["dataset_count"], 1117)
        self.assertEqual(audit["baseline_coverage"], 1117)
        self.assertEqual(audit["statistical_coverage"], 1117)
        self.assertEqual(audit["structural_coverage"], 1117)
        self.assertEqual(audit["change_point_coverage"], 1117)
        self.assertEqual(audit["reconstruction_coverage"], 251)
        self.assertEqual(audit["missing_override_coverage"], 20)
        self.assertEqual(audit["missing_datasets_without_override"], [])
        self.assertEqual(audit["source_contract"]["baseline"]["experiment"], "exp253")
        self.assertEqual(
            audit["source_contract"]["missing_override"],
            {
                "experiment": "exp260_262",
                "variant": "corrected_four_sources",
                "alpha": 0.005,
                "provenance_audit": "/Users/minho/Documents/Dataset/exp260_262_missing_aware_alert/audit.json",
            },
        )

    def test_select_rows_rejects_duplicate_dataset_for_source_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.jsonl"
            artifact.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"dataset": "demo", "variant": "fixed", "alpha": 0.005},
                        {"dataset": "demo", "variant": "fixed", "alpha": 0.005},
                    )
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate source row: demo"):
                select_rows(artifact, "fixed", 0.005)

    def test_select_rows_rejects_boolean_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.jsonl"
            artifact.write_text(
                json.dumps({"dataset": "demo", "variant": "fixed", "alpha": True})
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SourceContractError, "alpha is not canonical"):
                select_rows(artifact, "fixed", 0.005)

    def test_select_rows_requires_exact_alpha_without_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.jsonl"
            artifact.write_text(
                json.dumps(
                    {"dataset": "demo", "variant": "fixed", "alpha": 0.005000000001}
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(select_rows(artifact, "fixed", 0.005), {})

    def test_indices_reject_boolean_and_fractional_values(self) -> None:
        for value in (True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                SourceContractError, "non-integral index"
            ):
                _indices([value], source="unit", dataset="demo")

    def test_n_test_rejects_boolean_and_fractional_values(self) -> None:
        for value in (True, 3.5):
            with self.subTest(value=value), self.assertRaisesRegex(
                SourceContractError, "n_test is not an integer"
            ):
                _validate_n_test(
                    source="unit",
                    dataset="demo",
                    row={"n_test": value},
                    expected_n_test=3,
                )

    def test_row_indices_reject_non_list_index_field(self) -> None:
        with self.assertRaisesRegex(SourceContractError, "indices are not a list"):
            _validate_row_indices(
                source="unit",
                dataset="demo",
                n_test=3,
                row={"candidate_indices": "0"},
            )

    def test_required_indices_reject_missing_candidate_and_quality_fields(self) -> None:
        for field in ("candidate_indices", "quality_flag_indices"):
            with self.subTest(field=field), self.assertRaisesRegex(
                SourceContractError, f"missing required field: {field}"
            ):
                _required_indices({}, field=field, source="unit", dataset="demo")

    def test_schema_mismatch_is_a_contract_error(self) -> None:
        with self.assertRaisesRegex(SourceContractError, "source schema mismatch"):
            _validate_row_schema(
                source="unit",
                dataset="demo",
                row={"dataset": "demo"},
                schema={"dataset": str, "n_test": int},
            )

    def test_manifest_rejects_bad_hash_and_row_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "frozen_predictions.jsonl"
            artifact.write_text('{"dataset":"demo"}\n', encoding="utf-8")
            manifest = artifact.parent / "prediction_manifest.json"
            spec = config.SourceSpec(
                experiment="exp999",
                artifact_path=artifact,
                variant="fixed",
                alpha=0.005,
                scope="unit",
            )
            manifest.write_text(
                json.dumps(
                    {
                        "prediction_sha256": "wrong",
                        "rows": 2,
                        "datasets": 1,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SourceContractError, "SHA256 mismatch"):
                _validate_manifest("unit", spec, artifact_row_count=1, dataset_count=1)

            manifest.write_text(
                json.dumps(
                    {
                        "prediction_sha256": sha256_file(artifact),
                        "rows": 2,
                        "datasets": 1,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SourceContractError, "row count mismatch"):
                _validate_manifest("unit", spec, artifact_row_count=1, dataset_count=1)

    def test_missing_experiment_is_a_contract_error(self) -> None:
        with self.assertRaisesRegex(SourceContractError, "experiment missing"):
            _validate_experiment("baseline", config.SOURCE_CONTRACT["baseline"], {})

    def test_population_rejects_baseline_dataset_absent_from_database(self) -> None:
        metadata = {f"synthetic_{index}": (1, 1) for index in range(1117)}
        metadata.update(
            {
                "CornellWhaleChallenge": (1, 1),
                "Wafer_normal_1": (1, 1),
            }
        )
        with patch(
            "experiments.exp263_264_full_missing_aware_integration.inputs._database_metadata",
            return_value=metadata,
        ):
            with self.assertRaisesRegex(SourceContractError, "source dataset absent from DB"):
                load_population()

    def test_cli_returns_exit_2_for_contract_error(self) -> None:
        with patch(
            "experiments.exp263_264_full_missing_aware_integration.inputs.audit_source_contract",
            side_effect=SourceContractError("bad contract"),
        ), patch("sys.stderr", new=io.StringIO()):
            self.assertEqual(main(), 2)


if __name__ == "__main__":
    unittest.main()
