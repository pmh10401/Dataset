from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.exp403_408.contracts import DatasetInput, FeatureBlock, Fold
from experiments.exp403_408.feature_registry import (
    FEATURE_GROUPS,
    FeatureSource,
    load_feature_blocks,
    missing_feature_block,
    registry_for_names,
)
from experiments.exp403_408.feature_stability import (
    audit_feature_stability,
    build_group_summary,
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

    def test_cache_archive_label_or_metric_member_is_explicitly_unavailable(self) -> None:
        for member in (
            "test_labels",
            "test_accuracy",
            "test_auc",
            "candidate_tp",
            "candidate_fp",
            "candidate_f1_score",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = self._cache_archive(root)
                archive.parent.mkdir()
                np.savez_compressed(
                    archive,
                    train=np.asarray([0.1, 0.2, 0.3]),
                    test=np.asarray([0.4, 0.5]),
                    **{member: np.asarray([1.0])},
                )
                self._write_cache_metadata(root, archive)
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
            self.assertIsNone(block.train_oof)
            self.assertIsNone(block.test_values)

    def test_corrupt_cache_archive_is_explicitly_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = self._cache_archive(root)
            archive.parent.mkdir()
            np.savez_compressed(
                archive,
                train=np.asarray([0.1, 0.2, 0.3]),
                test=np.asarray([0.4, 0.5]),
            )
            archive.write_bytes(archive.read_bytes()[:-20])
            self._write_cache_metadata(root, archive)
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
        self.assertEqual(block.reason, "source_artifact_invalid")
        self.assertIsNone(block.train_oof)
        self.assertIsNone(block.test_values)

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

    def _write_cache_metadata(self, root: Path, archive: Path) -> None:
        archive.with_suffix(".json").write_text(
            json.dumps(
                {
                    "dataset": "toy",
                    "n_train": 3,
                    "n_test": 2,
                    "contains_test_labels": False,
                    "cache_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                }
            ),
            encoding="utf-8",
        )


class FeatureStabilityTests(unittest.TestCase):
    def test_constant_and_nonfinite_features_are_excluded(self) -> None:
        audit = audit_feature_stability(self._blocks_with_constant_and_nan(), self._folds())

        self.assertEqual(audit.excluded["constant"], "near_zero_train_variance")
        self.assertEqual(audit.excluded["nan"], "non_finite_train_oof")

    def test_correlated_features_collapse_by_train_oof_only(self) -> None:
        audit = audit_feature_stability(self._highly_correlated_blocks(), self._folds())

        self.assertEqual(audit.correlation_components, (("first", "second"),))
        self.assertEqual(audit.correlation_threshold, 0.90)
        self.assertEqual(audit.representatives, ("first", "third"))

    def test_unavailable_stale_source_does_not_set_train_oof_row_count(self) -> None:
        stale = FeatureBlock(
            name="stale",
            group="normal_space_distance",
            train_oof=np.asarray([1.0]),
            test_values=np.asarray([1.0]),
            source_hash="sha256:stale",
            available=False,
            reason="source_artifact_missing",
        )
        audit = audit_feature_stability((stale, self._block("usable", np.arange(8))), self._folds())

        self.assertEqual(audit.excluded["stale"], "source_artifact_missing")
        self.assertEqual(audit.representatives, ("usable",))

    def test_group_summary_uses_train_oof_not_test_values(self) -> None:
        blocks = self._highly_correlated_blocks()
        altered_test_values = tuple(
            FeatureBlock(
                name=block.name,
                group=block.group,
                train_oof=block.train_oof,
                test_values=np.asarray(block.test_values) * -1000.0,
                source_hash=block.source_hash,
            )
            for block in blocks
        )

        original = build_group_summary(audit_feature_stability(blocks, self._folds()))
        altered = build_group_summary(
            audit_feature_stability(altered_test_values, self._folds())
        )

        self.assertEqual(self._stable_hash(original), self._stable_hash(altered))

    def test_row_order_does_not_change_group_summary(self) -> None:
        rows = np.asarray([0.0, 3.0, 1.0, 5.0, 2.0, 7.0, 4.0, 6.0])

        original = build_group_summary(self._audit_for(rows))
        reversed_rows = build_group_summary(self._audit_for(rows[::-1]))

        self.assertEqual(self._stable_hash(original), self._stable_hash(reversed_rows))

    def _blocks_with_constant_and_nan(self) -> tuple[FeatureBlock, ...]:
        return (
            self._block("constant", [1.0] * 8),
            self._block("nan", [0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0]),
            self._block("usable", np.arange(8, dtype=float)),
        )

    def _highly_correlated_blocks(self) -> tuple[FeatureBlock, ...]:
        values = np.arange(8, dtype=float)
        return (
            self._block("first", values),
            self._block("second", values * 2.0 + 3.0),
            self._block("third", [0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 6.0, 5.0]),
        )

    def _audit_for(self, rows: np.ndarray):
        return audit_feature_stability(
            (
                self._block("primary", rows),
                self._block("secondary", rows**2 + 0.25),
            ),
            self._folds(),
        )

    def _folds(self) -> tuple[Fold, ...]:
        return (
            Fold(train_indices=(2, 3, 4, 5, 6, 7), validation_indices=(0, 1)),
            Fold(train_indices=(0, 1, 4, 5, 6, 7), validation_indices=(2, 3)),
            Fold(train_indices=(0, 1, 2, 3, 6, 7), validation_indices=(4, 5)),
            Fold(train_indices=(0, 1, 2, 3, 4, 5), validation_indices=(6, 7)),
        )

    def _block(self, name: str, train_oof: object) -> FeatureBlock:
        return FeatureBlock(
            name=name,
            group="normal_space_distance",
            train_oof=np.asarray(train_oof, dtype=float),
            test_values=np.asarray([9.0, 10.0]),
            source_hash=f"sha256:{name}",
        )

    def _stable_hash(self, value: object) -> str:
        payload = json.dumps(asdict(value), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    unittest.main()
