from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp267_273_zero_support_fn_recovery import config
from experiments.exp267_273_zero_support_fn_recovery import diagnostics as diagnostics_module
from experiments.exp267_273_zero_support_fn_recovery.diagnostics import (
    PopulationContractError,
    _read_exp266_csvs,
    build_zero_support_diagnostics,
    diagnostic_row,
    parse_dataset_family,
    verify_database_hash_against_summary,
    write_exp267_outputs,
)
from experiments.exp267_273_zero_support_fn_recovery.population import (
    EXP266_OUTPUT_ROOT,
    load_frozen_population,
    read_frozen_prediction_rows,
    validate_population,
)


class DiagnosticTests(unittest.TestCase):
    def test_population_requires_1117_unique_datasets(self) -> None:
        with self.assertRaises(PopulationContractError):
            validate_population(("a", "a"), expected=1117)

    def test_zero_support_is_computed_from_frozen_candidates(self) -> None:
        row = diagnostic_row(
            true_indices={1, 2},
            combined_indices={1},
            support={2: ()},
        )
        self.assertEqual(row["remaining_fn_indices"], [2])
        self.assertEqual(row["zero_support_fn_indices"], [2])
        self.assertEqual(row["single_source_fn_indices"], [])

    def test_single_source_is_kept_separate_from_zero_support(self) -> None:
        row = diagnostic_row(
            true_indices={1, 2, 3},
            combined_indices=set(),
            support={1: ("kernel",), 2: ("kernel", "statistical")},
        )
        self.assertEqual(row["zero_support_fn_indices"], [3])
        self.assertEqual(row["single_source_fn_indices"], [1])
        self.assertEqual(row["remaining_fn_count"], 3)

    def test_population_loader_checks_prediction_hash_before_returning_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "03_frozen_predictions.jsonl"
            prediction.write_text(
                json.dumps({"dataset": "b", "candidate_indices": [1]}) + "\n"
                + json.dumps({"dataset": "a", "candidate_indices": []}) + "\n",
                encoding="utf-8",
            )
            manifest = root / "04_prediction_manifest.json"
            manifest.write_text(
                json.dumps({"prediction_sha256": "wrong"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_frozen_population(prediction)

    def test_build_diagnostics_does_not_accept_prediction_rows_as_labels(self) -> None:
        prediction_rows = [
            {"dataset": "demo", "hard_indices": [], "support_by_index": {}}
        ]
        with self.assertRaises(ValueError):
            build_zero_support_diagnostics(
                prediction_rows=prediction_rows,
                result_rows=[{"dataset": "demo", "combined_fn": 1}],
                true_indices_by_dataset=None,
            )

    def test_public_runner_rejects_non_official_database_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "official DB_PATH"):
                write_exp267_outputs(database_path=Path(directory) / "test.db")

    def test_prediction_hash_and_parse_use_one_prediction_byte_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "03_frozen_predictions.jsonl"
            payload = (
                json.dumps({"dataset": "a", "candidate_indices": []}) + "\n"
            ).encode("utf-8")
            prediction.write_bytes(payload)
            (root / "04_prediction_manifest.json").write_text(
                json.dumps({
                    "prediction_sha256": hashlib.sha256(payload).hexdigest()
                }),
                encoding="utf-8",
            )
            original_read_bytes = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=lambda path: original_read_bytes(path),
            ) as read_bytes:
                rows, actual_hash, _ = read_frozen_prediction_rows(prediction)
            self.assertEqual(rows[0]["dataset"], "a")
            self.assertEqual(actual_hash, hashlib.sha256(payload).hexdigest())
            self.assertEqual(
                [call.args[0] for call in read_bytes.call_args_list], [prediction]
            )

    def test_csv_hash_and_parse_use_one_byte_buffer_per_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "05_dataset_results.csv").write_bytes(
                b"dataset,combined_fn\ndemo,1\n"
            )
            (root / "06_global_summary.csv").write_bytes(
                b"datasets,combined_fn\n1,1\n"
            )
            original_read_bytes = Path.read_bytes
            with patch.object(
                Path,
                "read_bytes",
                autospec=True,
                side_effect=lambda path: original_read_bytes(path),
            ) as read_bytes:
                results, summary, hashes = _read_exp266_csvs(root)
            self.assertEqual(results[0]["combined_fn"], "1")
            self.assertEqual(summary["datasets"], "1")
            self.assertEqual(
                [call.args[0] for call in read_bytes.call_args_list],
                [root / "05_dataset_results.csv", root / "06_global_summary.csv"],
            )
            self.assertEqual(
                hashes["05_dataset_results.csv"],
                hashlib.sha256((root / "05_dataset_results.csv").read_bytes()).hexdigest(),
            )

    def test_family_parser_uses_prefix_before_last_normal_marker(self) -> None:
        self.assertEqual(parse_dataset_family("ECG5000_normal_3"), "ECG5000")
        self.assertEqual(
            parse_dataset_family("family_normal_variant_normal_3"),
            "family_normal_variant",
        )
        self.assertEqual(parse_dataset_family("unmatched_dataset"), "unmatched_dataset")

    def test_live_outputs_record_db_hash_family_totals_and_frozen_totals(self) -> None:
        paths = write_exp267_outputs()
        with Path(paths["population_contract"]).open(encoding="utf-8") as handle:
            population_audit = json.load(handle)
        self.assertEqual(
            population_audit["database_sha256"],
            hashlib.sha256(config.DB_PATH.read_bytes()).hexdigest(),
        )
        with Path(paths["summary"]).open(newline="", encoding="utf-8") as handle:
            totals = next(csv.DictReader(handle))
        self.assertEqual(
            {key: int(totals[key]) for key in ("datasets", "combined_fn", "zero_support_fn", "single_source_fn")},
            {"datasets": 1117, "combined_fn": 1553, "zero_support_fn": 1405, "single_source_fn": 148},
        )
        family_path = paths["family_summary"]
        with family_path.open(newline="", encoding="utf-8") as handle:
            family_rows = list(csv.DictReader(handle))
        self.assertTrue(family_rows)
        self.assertTrue(all(row["diagnostic_only"] == "True" for row in family_rows))
        self.assertEqual(sum(int(row["combined_fn"]) for row in family_rows), 1553)
        self.assertEqual(sum(int(row["zero_support_fn"]) for row in family_rows), 1405)
        self.assertEqual(sum(int(row["single_source_fn"]) for row in family_rows), 148)

    def test_official_database_hash_must_match_frozen_exp266_summary(self) -> None:
        actual_hash = hashlib.sha256(config.DB_PATH.read_bytes()).hexdigest()
        self.assertEqual(
            verify_database_hash_against_summary(
                config.DB_PATH, {"database_sha256": actual_hash}
            ),
            actual_hash,
        )
        with self.assertRaisesRegex(ValueError, "database SHA256"):
            verify_database_hash_against_summary(
                config.DB_PATH, {"database_sha256": "0" * 64}
            )

    def test_database_hash_mismatch_fails_before_test_label_loading(self) -> None:
        original = diagnostics_module._read_exp266_csvs(EXP266_OUTPUT_ROOT)
        mismatched_summary = dict(original[1])
        mismatched_summary["database_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                diagnostics_module,
                "_read_exp266_csvs",
                return_value=(original[0], mismatched_summary, original[2]),
            ), patch.object(diagnostics_module, "load_test_truth") as load_labels:
                with self.assertRaisesRegex(ValueError, "database SHA256"):
                    diagnostics_module._write_exp267_outputs_with_database(
                        output_root=Path(directory),
                        database_path=config.DB_PATH,
                    )
            load_labels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
