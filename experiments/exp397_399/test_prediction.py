from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp397_399.prediction import (
    build_prediction,
    load_verified_ranking,
    write_ranking_artifact,
)


def exp292_fixture() -> list[dict[str, object]]:
    return [
        {
            "dataset": "A",
            "n_test": 4,
            "standard_review_indices": [1, 3],
            "priority_review_indices": [],
            "automatic_hard_indices": [0],
            "review_priority_1_indices": [1],
            "review_priority_2_indices": [],
            "review_priority_3_indices": [3],
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
    ]


def exp393_fixture() -> list[dict[str, object]]:
    return [
        {
            "dataset": "A",
            "n_test": 4,
            "p_values": {
                "transition": [1.0, 0.001, 1.0, 0.2],
                "dwell": [1.0, 0.2, 1.0, 0.3],
                "recovery": [1.0, 0.3, 1.0, 0.4],
                "composite": [1.0, 0.001, 1.0, 0.2],
            },
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
    ]


class PredictionTests(unittest.TestCase):
    def test_prediction_contains_no_test_metrics(self):
        rows = build_prediction(exp292_fixture(), exp393_fixture())

        forbidden = {"y_test", "tp", "fp", "precision", "f1"}
        self.assertTrue(all(forbidden.isdisjoint(row) for row in rows))
        self.assertTrue(all(row["contains_test_labels"] is False for row in rows))

    def test_every_variant_has_identical_candidate_universe(self):
        rows = build_prediction(exp292_fixture(), exp393_fixture())
        universes: dict[tuple[str, float], set[tuple[str, int]]] = {}
        for row in rows:
            key = (str(row["variant"]), float(row["alpha"]))
            universes.setdefault(key, set()).add(
                (str(row["dataset"]), int(row["candidate_index"]))
            )

        self.assertEqual(len({frozenset(value) for value in universes.values()}), 1)
        self.assertEqual(next(iter(universes.values())), {("A", 1), ("A", 3)})

    def test_prediction_artifact_rejects_hash_tampering(self):
        rows = build_prediction(exp292_fixture(), exp393_fixture())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path, manifest_path = write_ranking_artifact(root, rows, {})
            prediction_path.write_text(
                prediction_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_ranking(prediction_path, manifest_path)

    def test_prediction_rows_are_sorted_deterministically(self):
        rows = build_prediction(exp292_fixture(), exp393_fixture())
        with tempfile.TemporaryDirectory() as directory:
            prediction_path, manifest_path = write_ranking_artifact(
                Path(directory), rows, {}
            )
            loaded, manifest = load_verified_ranking(
                prediction_path, manifest_path
            )

        self.assertEqual(len(loaded), 32)
        self.assertEqual(manifest["candidate_universe_count"], 2)
        self.assertEqual(
            json.dumps(loaded, sort_keys=True),
            json.dumps(sorted(loaded, key=lambda row: (
                row["variant"],
                row["alpha"],
                row["rank"],
            )), sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
