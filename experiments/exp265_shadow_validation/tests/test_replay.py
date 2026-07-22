from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp265_shadow_validation.replay import build_replay_payload, load_source_confidence


class ReplayPayloadTest(unittest.TestCase):
    @staticmethod
    def _all_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            keys = {str(key).lower() for key in value}
            for item in value.values():
                keys.update(ReplayPayloadTest._all_keys(item))
            return keys
        if isinstance(value, (list, tuple)):
            keys: set[str] = set()
            for item in value:
                keys.update(ReplayPayloadTest._all_keys(item))
            return keys
        return set()

    def test_builds_label_free_replay_with_candidate_specific_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "fixture.db"
            with sqlite3.connect(db_path) as connection:
                connection.executescript(
                    """
                    create table datasets (
                      id integer primary key, name text, has_missing integer,
                      train_normal_count integer, test_total_count integer
                    );
                    create table instances (
                      dataset_id integer, split text, instance_index integer,
                      label text, values_blob blob
                    );
                    """
                )
                connection.execute(
                    "insert into datasets values (1, 'Demo_normal_0', 0, 120, 3)"
                )
                for index, values in enumerate(([0, 1, 0], [5, 6, 5], [2, 3, 2])):
                    connection.execute(
                        "insert into instances values (1, 'TEST', ?, ?, ?)",
                        (index, f"hidden-{index}", np.asarray(values, dtype=np.float32).tobytes()),
                    )
            row = {
                "dataset": "Demo_normal_0", "n_train": 120, "n_test": 3,
                "baseline_hard_indices": [1], "baseline_standard_indices": [2],
                "baseline_priority_indices": [], "final_hard_indices": [1],
                "final_standard_indices": [2], "final_priority_indices": [],
                "quality_flag_indices": [], "quality_flag_source": "verified_nonmissing_empty",
                "source_indices": {
                    "statistical": [1, 2], "structural": [1, 2],
                    "change_point": [1, 2], "reconstruction": [],
                },
            }
            confidence = {
                name: np.asarray([0.1, 0.999, 0.998], dtype=float)
                for name in ("statistical", "structural", "change_point")
            }
            confidence["reconstruction"] = None

            payload = build_replay_payload(row, db_path=db_path, confidence=confidence)

            self.assertEqual(payload["collection_mode"], "retrospective_replay")
            self.assertEqual(payload["policy_version"], "exp265-replay-v1")
            self.assertFalse(payload["contains_test_labels"])
            self.assertFalse(payload["contains_evaluation_metrics"])
            self.assertTrue(
                self._all_keys(payload).isdisjoint(
                    {"y_test", "true_label", "test_label", "hard_tp", "hard_fp", "hard_f1", "combined_f1"}
                )
            )
            self.assertEqual(payload["hard_indices"], [1])
            candidate = next(item for item in payload["candidates"] if item["candidate_index"] == 1)
            self.assertEqual(candidate["series_preview"]["values"], [5.0, 6.0, 5.0])

    def test_replay_module_never_selects_test_labels(self) -> None:
        source = Path(__file__).parents[1].joinpath("replay.py").read_text(encoding="utf-8")
        self.assertNotIn("select i.label", source.lower())
        self.assertNotIn("test_anomaly", source.lower())

    def test_empty_historical_score_cache_is_preserved_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            roots = {}
            key = "7849c4b680b7038065aa"
            for source in ("statistical", "structural", "change_point", "reconstruction"):
                root = Path(directory) / source
                root.mkdir()
                roots[source] = root
            np.savez(
                roots["statistical"] / f"{key}.npz",
                combined_40__train_oof=np.asarray([], dtype=float),
                combined_40__test_scores=np.asarray([], dtype=float),
            )
            result = load_source_confidence("Demo_normal_0", cache_roots=roots)
            self.assertIsNone(result["statistical"])
            self.assertIsNone(result["structural"])
            self.assertIsNone(result["change_point"])
            self.assertIsNone(result["reconstruction"])


if __name__ == "__main__":
    unittest.main()
