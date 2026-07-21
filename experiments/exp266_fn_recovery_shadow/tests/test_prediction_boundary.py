from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType
from unittest.mock import patch

from experiments.exp266_fn_recovery_shadow import prediction as prediction_module
from experiments.exp266_fn_recovery_shadow.config import (
    EVIDENCE_GROUPS,
    EXPECTED_DATASETS,
    FROZEN_PREDICTION_HASHES,
)
from experiments.exp266_fn_recovery_shadow.contracts import (
    FrozenDatasetInputs,
    FrozenRecoveryInputs,
)
from experiments.exp266_fn_recovery_shadow.inputs import InputContractError, sha256_file
from experiments.exp266_fn_recovery_shadow.prediction import freeze_rows, predict_all
from experiments.exp266_fn_recovery_shadow.routing import route_recovery


def prediction_row(dataset: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "hard_indices": [0],
        "shadow_indices": [],
        "priority_indices": [],
        "standard_indices": [],
        "recovery_shadow_indices": [1],
        "recovery_priority_indices": [],
        "data_quality_review_indices": [],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def example_inputs_with_n_test(n_test: int) -> FrozenDatasetInputs:
    return FrozenDatasetInputs(
        dataset="demo",
        n_train=100,
        n_test=n_test,
        hard=frozenset({0}),
        shadow=frozenset(),
        priority=frozenset(),
        standard=frozenset(),
        evidence={
            "kernel": frozenset({1}),
            "statistical": frozenset({1}),
            "structural": frozenset({1}),
            "change_point": frozenset(),
            "reconstruction": frozenset(),
        },
        available_groups=frozenset(EVIDENCE_GROUPS),
        quality_flags=frozenset(),
    )


def example_recovery_inputs(count: int) -> FrozenRecoveryInputs:
    population = tuple(f"dataset_{index:04d}" for index in range(count))
    datasets = {
        dataset: FrozenDatasetInputs(
            dataset=dataset,
            n_train=100,
            n_test=8,
            hard=frozenset({0}),
            shadow=frozenset(),
            priority=frozenset(),
            standard=frozenset(),
            evidence={group: frozenset() for group in EVIDENCE_GROUPS},
            available_groups=frozenset(EVIDENCE_GROUPS),
            quality_flags=frozenset(),
        )
        for dataset in population
    }
    return FrozenRecoveryInputs(
        population=population,
        datasets=MappingProxyType(datasets),
        input_sha256=MappingProxyType(dict(FROZEN_PREDICTION_HASHES)),
        lane_totals=MappingProxyType({"hard": count}),
    )


def smoke_manifest_extra(rows: int) -> dict[str, object]:
    return {
        "checkpoint": prediction_module.PREDICTION_FILENAME,
        "complete_population": False,
        "config": prediction_module._config_snapshot(),
        "config_sha256": hashlib.sha256(
            prediction_module._canonical_json(prediction_module._config_snapshot())
        ).hexdigest(),
        "coverage": {
            "expected_datasets": EXPECTED_DATASETS,
            "frozen_population": EXPECTED_DATASETS,
            "predicted_datasets": rows,
        },
        "deployment_eligible": False,
        "input_sha256": dict(FROZEN_PREDICTION_HASHES),
        "lane_policy": {
            "hard": {"autonomous": True, "unchanged": True},
            "recovery_shadow": {
                "deployment_eligible": False,
                "human_assisted": True,
                "review_only": True,
            },
            "recovery_priority": {
                "deployment_eligible": False,
                "human_assisted": True,
                "review_only": True,
            },
            "data_quality_review": {
                "deployment_eligible": False,
                "human_assisted": True,
                "review_only": True,
            },
        },
        "smoke": True,
    }


class PredictionBoundaryTests(unittest.TestCase):
    def test_prediction_has_no_evaluation_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = freeze_rows(Path(directory), [prediction_row("a")])
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            forbidden = {"y_test", "tp", "fp", "fn", "precision", "recall", "f1"}
            self.assertFalse(forbidden.intersection(key.lower() for key in row))
            self.assertIs(row["contains_test_labels"], False)

    def test_hash_is_input_order_invariant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = freeze_rows(root / "left", [prediction_row("b"), prediction_row("a")])
            right = freeze_rows(root / "right", [prediction_row("a"), prediction_row("b")])
            self.assertEqual(sha256_file(left), sha256_file(right))

    def test_appending_unselected_test_rows_keeps_prefix_routing(self) -> None:
        short = route_recovery(example_inputs_with_n_test(8))
        long = route_recovery(example_inputs_with_n_test(12))
        self.assertEqual(short.recovery_shadow, long.recovery_shadow)

    def test_prediction_api_has_no_label_parameter(self) -> None:
        self.assertNotIn("y_test", inspect.signature(predict_all).parameters)

    def test_freeze_rows_rejects_nested_forbidden_prediction_fields(self) -> None:
        row = prediction_row("a")
        row["support_sources"] = {"kernel": {"nested": {"label": [1]}}}
        with TemporaryDirectory() as directory, self.assertRaises(InputContractError):
            freeze_rows(Path(directory), [row])

    def test_freeze_rows_rejects_true_evaluation_or_anomaly_flags(self) -> None:
        for field in ("contains_evaluation_metrics", "has_anomaly_positions"):
            with self.subTest(field=field):
                row = prediction_row("a")
                row[field] = True
                with TemporaryDirectory() as directory, self.assertRaises(InputContractError):
                    freeze_rows(Path(directory), [row])

    def test_freeze_rows_rejects_test_length_at_any_depth(self) -> None:
        row = prediction_row("a")
        row["support_sources"] = {"kernel": {"n_test": 8}}
        with TemporaryDirectory() as directory, self.assertRaises(InputContractError):
            freeze_rows(Path(directory), [row])

    @patch("experiments.exp266_fn_recovery_shadow.prediction.load_frozen_inputs")
    def test_full_and_smoke_outputs_are_separate_and_explicit(self, load_inputs) -> None:
        load_inputs.return_value = example_recovery_inputs(EXPECTED_DATASETS)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            full_path = predict_all(root)
            full_manifest = json.loads(
                (root / "04_prediction_manifest.json").read_text(encoding="utf-8")
            )
            full_row = json.loads(full_path.read_text(encoding="utf-8").splitlines()[0])

            smoke_path = predict_all(root, dataset_limit=2)
            smoke_root = root / "smoke" / "dataset_limit_2"
            smoke_manifest = json.loads(
                (smoke_root / "04_prediction_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(full_path, root / "03_frozen_predictions.jsonl")
            self.assertEqual(smoke_path, smoke_root / "03_frozen_predictions.jsonl")
            self.assertNotEqual(full_path, smoke_path)
            self.assertNotIn("n_test", full_row)
            self.assertIs(full_manifest["complete_population"], True)
            self.assertIs(full_manifest["smoke"], False)
            self.assertIs(smoke_manifest["complete_population"], False)
            self.assertIs(smoke_manifest["smoke"], True)
            self.assertIs(smoke_manifest["deployment_eligible"], False)
            self.assertEqual(full_manifest["lane_policy"]["hard"], {"autonomous": True, "unchanged": True})
            for lane in ("recovery_shadow", "recovery_priority", "data_quality_review"):
                self.assertEqual(
                    full_manifest["lane_policy"][lane],
                    {"deployment_eligible": False, "human_assisted": True, "review_only": True},
                )

    @patch("experiments.exp266_fn_recovery_shadow.prediction.load_frozen_inputs")
    def test_full_prediction_rejects_incomplete_population(self, load_inputs) -> None:
        load_inputs.return_value = example_recovery_inputs(EXPECTED_DATASETS - 1)
        with TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "1,117"):
            predict_all(Path(directory))

    @patch("experiments.exp266_fn_recovery_shadow.prediction.load_frozen_inputs")
    def test_predict_rejects_non_partial_smoke_limits(self, load_inputs) -> None:
        load_inputs.return_value = example_recovery_inputs(EXPECTED_DATASETS)
        with TemporaryDirectory() as directory:
            for limit in (-1, True, "2", EXPECTED_DATASETS, EXPECTED_DATASETS + 1):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    predict_all(Path(directory), dataset_limit=limit)

    def test_pending_checkpoint_is_rejected_until_recovered(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "smoke" / "dataset_limit_1"
            freeze_rows(root, [prediction_row("a")], manifest_extra=smoke_manifest_extra(1))
            manifest = json.loads(
                (root / "04_prediction_manifest.json").read_text(encoding="utf-8")
            )
            (root / prediction_module.PENDING_FILENAME).write_text(
                json.dumps(
                    {
                        "generation": manifest["generation"],
                        "prediction_sha256": manifest["prediction_sha256"],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(prediction_module.CheckpointError):
                prediction_module.verify_prediction_checkpoint(root)

            recovered = prediction_module.recover_prediction_checkpoint(root)
            self.assertEqual(recovered["generation"], manifest["generation"])
            self.assertFalse((root / prediction_module.PENDING_FILENAME).exists())
            self.assertEqual(
                prediction_module.verify_prediction_checkpoint(root)["prediction_sha256"],
                manifest["prediction_sha256"],
            )

    def test_checkpoint_verification_requires_manifest_hash_match(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_rows(root, [prediction_row("a")])
            manifest_path = root / "04_prediction_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prediction_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(prediction_module.CheckpointError):
                prediction_module.verify_prediction_checkpoint(root)

    def test_verifier_rejects_direct_freeze_without_prediction_contract(self) -> None:
        with TemporaryDirectory() as directory, self.assertRaises(prediction_module.CheckpointError):
            root = Path(directory)
            freeze_rows(root, [prediction_row("a")])
            prediction_module.verify_prediction_checkpoint(root)

    @patch("experiments.exp266_fn_recovery_shadow.prediction.load_frozen_inputs")
    def test_verifier_rejects_missing_or_inconsistent_manifest_contract(self, load_inputs) -> None:
        load_inputs.return_value = example_recovery_inputs(EXPECTED_DATASETS)
        mutations = {
            "missing_coverage": lambda manifest: manifest.pop("coverage"),
            "inconsistent_coverage": lambda manifest: manifest["coverage"].update(
                {"predicted_datasets": EXPECTED_DATASETS - 1}
            ),
            "inconsistent_smoke": lambda manifest: manifest.update({"smoke": True}),
            "deployment_eligible": lambda manifest: manifest.update({"deployment_eligible": True}),
            "lane_policy": lambda manifest: manifest.update({"lane_policy": {}}),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    predict_all(root)
                    manifest_path = root / "04_prediction_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    mutate(manifest)
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(prediction_module.CheckpointError):
                        prediction_module.verify_prediction_checkpoint(root)

    @patch("experiments.exp266_fn_recovery_shadow.prediction.load_frozen_inputs")
    def test_verifier_rejects_smoke_location_mismatch(self, load_inputs) -> None:
        load_inputs.return_value = example_recovery_inputs(EXPECTED_DATASETS)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            predict_all(root, dataset_limit=2)
            smoke_root = root / "smoke" / "dataset_limit_2"
            wrong_root = root / "smoke" / "wrong"
            smoke_root.rename(wrong_root)
            with self.assertRaises(prediction_module.CheckpointError):
                prediction_module.verify_prediction_checkpoint(wrong_root)
