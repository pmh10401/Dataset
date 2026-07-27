from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp403_408.contracts import DatasetInput
from experiments.exp403_408.feature_registry import (
    FEATURE_GROUPS,
    FeatureSource,
    load_feature_blocks,
    missing_feature_block,
    registry_for_names,
)


class FeatureRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = DatasetInput(
            dataset="toy",
            train_normal=np.zeros((3, 4), dtype=float),
            test_values=np.ones((2, 4), dtype=float),
            target_length=4,
        )

    def test_registry_defines_exactly_six_physical_groups(self) -> None:
        self.assertEqual(
            FEATURE_GROUPS,
            {
                "temporal_random_kernel",
                "normal_space_distance",
                "change_delta",
                "frequency_multiscale",
                "shape_complexity",
                "process_grammar",
            },
        )

    def test_rocket_capacities_share_one_physical_group(self) -> None:
        blocks = registry_for_names(["rocket256", "rocket512", "minirocket1024"])
        self.assertEqual(
            {block.group for block in blocks}, {"temporal_random_kernel"}
        )

    def test_missing_source_is_explicit_not_empty_candidates(self) -> None:
        block = missing_feature_block("wavelet")
        self.assertFalse(block.available)
        self.assertEqual(block.reason, "source_artifact_missing")
        self.assertIsNone(block.train_oof)
        self.assertIsNone(block.test_values)

    def test_cache_adapter_returns_hash_verified_train_oof_and_test_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(
                archive,
                train=np.asarray([0.1, 0.2, 0.3]),
                test=np.asarray([0.4, 0.5]),
            )
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": False,
                        "seed": 7,
                        "cache_sha256": archive_hash,
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
                seed_values=(("seed", 7),),
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertTrue(block.available)
        self.assertEqual(block.source_hash, archive_hash)
        np.testing.assert_allclose(block.train_oof, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(block.test_values, [0.4, 0.5])

    def test_cache_contract_mismatch_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(archive, train=np.asarray([0.1, 0.2, 0.3]), test=np.asarray([0.4, 0.5]))
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "different_dataset",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": False,
                        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertFalse(block.available)
        self.assertEqual(block.reason, "source_dataset_mismatch")
        self.assertIsNone(block.train_oof)
        self.assertIsNone(block.test_values)

    def test_label_derived_source_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(archive, train=np.asarray([0.1, 0.2, 0.3]), test=np.asarray([0.4, 0.5]))
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": True,
                        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertFalse(block.available)
        self.assertEqual(block.reason, "source_label_derived")

    def test_undeclared_test_metric_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(archive, train=np.asarray([0.1, 0.2, 0.3]), test=np.asarray([0.4, 0.5]))
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": False,
                        "test_accuracy": 1.0,
                        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertFalse(block.available)
        self.assertEqual(block.reason, "source_label_derived")

    def test_undeclared_label_metric_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(archive, train=np.asarray([0.1, 0.2, 0.3]), test=np.asarray([0.4, 0.5]))
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": False,
                        "candidate_f1_score": 1.0,
                        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertFalse(block.available)
        self.assertEqual(block.reason, "source_label_derived")

    def test_cache_adapter_verifies_a_present_seed_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(archive, train=np.asarray([0.1, 0.2, 0.3]), test=np.asarray([0.4, 0.5]))
            archive.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "dataset": "toy",
                        "n_train": 3,
                        "n_test": 2,
                        "contains_test_labels": False,
                        "config": {"seeds": [7, 8]},
                        "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="distance",
                group="normal_space_distance",
                kind="cache",
                root=root,
                train_key="train",
                test_key="test",
                seed_values=(("config.seeds", (7, 8)),),
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertTrue(block.available)

    def test_grammar_adapter_verifies_manifest_seed_not_prediction_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "predictions.jsonl"
            row = {
                "dataset": "toy",
                "n_train_normal": 3,
                "n_test": 2,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "test_metrics_select_policy": False,
                "abstained": False,
                "train_oof_scores": {"composite": [0.1, 0.2, 0.3]},
                "scores": {"composite": [0.4, 0.5]},
            }
            prediction.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                        "config": {"random_seed": 7},
                        "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            source = FeatureSource(
                name="grammar",
                group="process_grammar",
                kind="grammar",
                root=root,
                train_key="composite",
                test_key="composite",
                seed_values=(("config.random_seed", 7),),
                prediction_name="predictions.jsonl",
                manifest_name="manifest.json",
            )

            (block,) = load_feature_blocks("toy", self.bundle, sources=(source,))

        self.assertTrue(block.available)

    def _cache_archive(self, root: Path) -> Path:
        stem = hashlib.sha256(b"toy").hexdigest()[:20]
        return root / "cache" / f"{stem}.npz"


if __name__ == "__main__":
    unittest.main()
