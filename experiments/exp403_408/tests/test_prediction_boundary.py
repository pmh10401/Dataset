from __future__ import annotations

import dataclasses
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp403_408.contracts import ALPHAS, DatasetInput, FrozenConfig, PredictionRow
from experiments.exp403_408.prediction import (
    FEATURE_MANIFEST_NAME,
    HARD_CONFIDENCE_NAME,
    PREDICTION_MANIFEST_NAME,
    find_forbidden_fields,
    predict_dataset,
    verify_prediction,
    write_prediction,
)


def bundle() -> DatasetInput:
    return DatasetInput(
        dataset="toy",
        train_normal=np.asarray([[0.0, 0.0], [0.2, 0.2], [0.4, 0.4]], dtype=float),
        test_values=np.asarray([[0.1, 0.1], [0.9, 0.9]], dtype=float),
        target_length=2,
        hard_indices=frozenset({1}),
    )


def config() -> FrozenConfig:
    return FrozenConfig(target_length=2)


def valid_rows() -> list[PredictionRow]:
    return [
        PredictionRow(
            dataset="zeta",
            candidate_index=2,
            scores_by_alpha={str(alpha): 0.5 for alpha in ALPHAS},
            confidence_by_alpha={str(alpha): "recheck" for alpha in ALPHAS},
            eligibility="abstain",
            abstention_reason="insufficient_train_normal",
        ),
        PredictionRow(
            dataset="alpha",
            candidate_index=1,
            scores_by_alpha={str(alpha): 0.25 for alpha in ALPHAS},
            confidence_by_alpha={str(alpha): "high" for alpha in ALPHAS},
            eligibility="classical_committee",
        ),
    ]


class PredictionBoundaryTests(unittest.TestCase):
    def test_predict_dataset_signature_has_no_y_test(self) -> None:
        self.assertNotIn("y_test", inspect.signature(predict_dataset).parameters)

    def test_prediction_json_has_no_evaluation_fields(self) -> None:
        payload = dataclasses.asdict(predict_dataset(bundle(), config()))
        self.assertFalse(find_forbidden_fields(payload))

    def test_modified_prediction_hash_stops_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_prediction(valid_rows(), output)
            output.joinpath(HARD_CONFIDENCE_NAME).write_text("modified", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_prediction(output)

    def test_writes_canonical_artifacts_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_prediction(valid_rows(), output, config=config())
            rows, verified_manifest = verify_prediction(output)

            self.assertEqual([row["dataset"] for row in rows], ["alpha", "zeta"])
            self.assertEqual(
                manifest["artifact_hashes"],
                verified_manifest["artifact_hashes"],
            )
            self.assertEqual(
                set(verified_manifest["artifact_hashes"]),
                {"config", "input", "feature", "prediction", "code", "baseline"},
            )
            self.assertTrue(output.joinpath(FEATURE_MANIFEST_NAME).is_file())
            self.assertTrue(output.joinpath(HARD_CONFIDENCE_NAME).is_file())
            self.assertTrue(output.joinpath(PREDICTION_MANIFEST_NAME).is_file())
            self.assertEqual(
                [json.loads(line)["dataset"] for line in output.joinpath(HARD_CONFIDENCE_NAME).read_text(encoding="utf-8").splitlines()],
                ["alpha", "zeta"],
            )


if __name__ == "__main__":
    unittest.main()
