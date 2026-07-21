from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.diagnostics import (
    PopulationContractError,
    build_zero_support_diagnostics,
    diagnostic_row,
)
from experiments.exp267_273_zero_support_fn_recovery.population import (
    load_frozen_population,
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


if __name__ == "__main__":
    unittest.main()
