from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.prediction import (
    _variants,
    build_prediction_row,
    predict_source,
    verify_file_hash,
)
from experiments.exp239_248_diverse_score_sources.registry import SOURCE_REGISTRY


class PipelineContractTest(unittest.TestCase):
    def test_registry_has_exact_experiment_order(self) -> None:
        self.assertEqual(tuple(SOURCE_REGISTRY), tuple(range(239, 249)))

    def test_prediction_rows_have_no_evaluation_fields(self) -> None:
        row = build_prediction_row(
            experiment=239,
            source="source",
            variant="default",
            dataset="dataset",
            alpha=0.005,
            n_test=5,
            candidates={1, 3},
            hard={1},
            standard={2},
            priority={4},
            abstained=False,
            resolution_blocked=False,
            attainable_p=0.001,
        )
        forbidden = {"y_test", "labels", "tp", "fp", "f1", "actual_anomaly_positions"}
        self.assertFalse(forbidden & set(row))

    def test_prediction_rows_normalize_numpy_scalar_indices(self) -> None:
        row = build_prediction_row(
            experiment=239,
            source="source",
            variant="default",
            dataset="dataset",
            alpha=np.float64(0.01),
            n_test=np.int64(5),
            candidates={np.int64(1)},
            hard={np.int64(1)},
            standard=set(),
            priority=set(),
            abstained=np.bool_(False),
            resolution_blocked=np.bool_(False),
            attainable_p=np.float64(0.01),
        )
        json.dumps(row)

    def test_prediction_code_cannot_read_labels(self) -> None:
        source = inspect.getsource(predict_source).lower()
        self.assertNotIn("load_test_labels", source)
        self.assertNotIn("y_test", inspect.signature(predict_source).parameters)

    def test_hash_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.jsonl"
            path.write_text(json.dumps({"value": 1}) + "\n")
            expected = verify_file_hash(path)
            path.write_text(json.dumps({"value": 2}) + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_file_hash(path, expected)

    def test_abstained_model_rows_keep_all_registered_variants(self) -> None:
        archive = {"train_oof": np.asarray([]), "test_scores": np.asarray([])}
        self.assertEqual(set(_variants(246, archive)), {"default", "seed_intersection"})
        self.assertEqual(set(_variants(247, archive)), {"default", "seed_intersection"})
        self.assertEqual(set(_variants(248, archive)), {"raw_nll", "typicality"})


if __name__ == "__main__":
    unittest.main()
