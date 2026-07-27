from __future__ import annotations

import dataclasses
import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp403_408.contracts import ALPHAS, DatasetInput, FrozenConfig, PredictionRow
from experiments.exp403_408.artifact_io import canonical_json_bytes, sha256_bytes, sha256_file
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


def frozen_hard_lanes() -> dict[str, tuple[int, ...]]:
    return {"alpha": (1,), "zeta": (2,)}


def artifact_provenance(root: Path) -> dict[str, dict[str, Path]]:
    sources = root / "sources"
    sources.mkdir()
    artifacts = {
        "feature_artifacts": {"exp403_features": sources / "exp403_features.npz"},
        "input_artifacts": {"exp291_lanes": sources / "exp291_lanes.jsonl"},
        "baseline_artifacts": {"exp292_baseline": sources / "exp292_baseline.jsonl"},
        "config_artifacts": {"frozen_config": sources / "config.json"},
        "code_artifacts": {"exp404_committee": sources / "classical_committee.py"},
    }
    for category in artifacts.values():
        for artifact in category.values():
            artifact.write_text(artifact.name, encoding="utf-8")
    return artifacts


class PredictionBoundaryTests(unittest.TestCase):
    def test_predict_dataset_signature_has_no_y_test(self) -> None:
        self.assertNotIn("y_test", inspect.signature(predict_dataset).parameters)

    def test_prediction_json_has_no_evaluation_fields(self) -> None:
        payload = [dataclasses.asdict(row) for row in predict_dataset(bundle(), config())]
        self.assertFalse(find_forbidden_fields(payload))

    def test_predict_dataset_preserves_every_frozen_hard_candidate(self) -> None:
        multi_hard_bundle = dataclasses.replace(bundle(), hard_indices=frozenset({0, 1}))

        rows = predict_dataset(multi_hard_bundle, config())

        self.assertEqual([row.candidate_index for row in rows], [0, 1])

    def test_write_and_verify_require_exact_frozen_hard_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            provenance = artifact_provenance(Path(directory))
            incomplete = valid_rows()[:-1]

            with self.assertRaisesRegex(ValueError, "frozen Hard.*coverage"):
                write_prediction(
                    incomplete,
                    output,
                    config=config(),
                    frozen_hard_lanes=frozen_hard_lanes(),
                    **provenance,
                )

    def test_modified_prediction_hash_stops_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            write_prediction(
                valid_rows(),
                output,
                config=config(),
                frozen_hard_lanes=frozen_hard_lanes(),
                **artifact_provenance(root),
            )
            output.joinpath(HARD_CONFIDENCE_NAME).write_text("modified", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_prediction(output)

    def test_writes_canonical_artifacts_and_manifest_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            manifest = write_prediction(
                valid_rows(),
                output,
                config=config(),
                frozen_hard_lanes=frozen_hard_lanes(),
                **artifact_provenance(root),
            )
            rows, verified_manifest = verify_prediction(output)

            self.assertEqual([row["dataset"] for row in rows], ["alpha", "zeta"])
            self.assertEqual(
                manifest["artifact_hashes"],
                verified_manifest["artifact_hashes"],
            )
            self.assertEqual(
                set(verified_manifest["artifact_hashes"]),
                {"config", "input", "feature", "prediction", "code", "baseline", "feature_manifest"},
            )
            self.assertTrue(output.joinpath(FEATURE_MANIFEST_NAME).is_file())
            self.assertTrue(output.joinpath(HARD_CONFIDENCE_NAME).is_file())
            self.assertTrue(output.joinpath(PREDICTION_MANIFEST_NAME).is_file())
            self.assertEqual(
                verified_manifest["provenance"]["code_artifacts"]["prediction_boundary"]["path"],
                str(Path(__file__).parents[1] / "prediction.py"),
            )
            self.assertEqual(
                [json.loads(line)["dataset"] for line in output.joinpath(HARD_CONFIDENCE_NAME).read_text(encoding="utf-8").splitlines()],
                ["alpha", "zeta"],
            )

    def test_rejects_nested_recommendation_and_evaluation_fields(self) -> None:
        for key in ("recommendation", "metrics", "accuracy"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                row = dataclasses.asdict(valid_rows()[0])
                row["nested"] = {key: "review"}

                with self.assertRaisesRegex(ValueError, "forbidden"):
                    write_prediction(
                        [row, valid_rows()[1]],
                        output,
                        config=config(),
                        frozen_hard_lanes=frozen_hard_lanes(),
                        **artifact_provenance(root),
                    )

    def test_requires_non_empty_actual_artifact_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "provenance"):
                write_prediction(
                    valid_rows(),
                    Path(directory) / "output",
                    config=config(),
                    frozen_hard_lanes=frozen_hard_lanes(),
                    feature_artifacts={},
                    input_artifacts={},
                    baseline_artifacts={},
                    config_artifacts={},
                    code_artifacts={},
                )

    def test_modified_provenance_artifact_stops_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            provenance = artifact_provenance(root)
            write_prediction(
                valid_rows(),
                output,
                config=config(),
                frozen_hard_lanes=frozen_hard_lanes(),
                **provenance,
            )
            provenance["feature_artifacts"]["exp403_features"].write_text("modified", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_prediction(output)

    def test_verified_prediction_rejects_rehashed_nested_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            write_prediction(
                valid_rows(), output, config=config(), frozen_hard_lanes=frozen_hard_lanes(), **artifact_provenance(root)
            )
            prediction_path = output / HARD_CONFIDENCE_NAME
            rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["nested"] = {"recommendation": "review"}
            prediction_path.write_text(
                "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = output / PREDICTION_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_hashes"]["prediction"] = sha256_file(prediction_path)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "forbidden"):
                verify_prediction(output)

    def test_verified_prediction_rejects_forged_frozen_hard_lane_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            write_prediction(
                valid_rows(), output, config=config(), frozen_hard_lanes=frozen_hard_lanes(), **artifact_provenance(root)
            )
            manifest_path = output / PREDICTION_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["frozen_hard_lanes"] = {"alpha": [], "zeta": [2]}
            manifest["frozen_hard_lanes_sha256"] = sha256_bytes(
                canonical_json_bytes(manifest["frozen_hard_lanes"])
            )
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frozen Hard.*coverage"):
                verify_prediction(output)

    def test_manifest_config_is_hash_bound_to_config_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            write_prediction(
                valid_rows(), output, config=config(), frozen_hard_lanes=frozen_hard_lanes(), **artifact_provenance(root)
            )
            manifest_path = output / PREDICTION_MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["config"]["target_length"] = 3
            manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "config SHA256"):
                verify_prediction(output)


if __name__ == "__main__":
    unittest.main()
