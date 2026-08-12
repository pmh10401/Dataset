"""Focused behavioral contracts for the label-free hard score sources."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset


def _raw_fixture(*, train_count: int = 5) -> RawDataset:
    grid = np.linspace(0.0, 2.0 * np.pi, 32, dtype=np.float32)
    train = tuple((np.sin(grid + index / 11.0) + index / 100.0).astype(np.float32)
                  for index in range(train_count))
    test = tuple((np.cos(grid + index / 7.0) + index / 80.0).astype(np.float32)
                 for index in range(3))
    return RawDataset(
        dataset="fixture",
        metadata_length=32,
        train_indices=tuple(range(len(train))),
        train_blobs=tuple(row.astype("<f4").tobytes() for row in train),
        train_rows=train,
        test_indices=tuple(range(len(test))),
        test_blobs=tuple(row.astype("<f4").tobytes() for row in test),
        test_rows=test,
    )


class HardSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = _raw_fixture()

    def test_source_bundle_is_deterministic_full_length_and_label_free(self) -> None:
        """Dropping, reordering, or TEST-label-fitting a score source breaks this boundary."""
        sys.modules.pop("paper.equipment_anomaly_detection.label_access_paper", None)
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
            EXPECTED_HARD_SOURCES,
            score_hard_sources,
        )

        first = score_hard_sources(self.values)
        second = score_hard_sources(self.values)
        self.assertEqual(first.source_names, EXPECTED_HARD_SOURCES)
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
        for name in first.source_names:
            source = first.by_name[name]
            self.assertIn(source.origin.node_id, {name, "exp84_family_guard"})
            if source.eligible:
                self.assertEqual(source.test_score.shape, (len(self.values.test_indices),))
            else:
                self.assertEqual(source.test_score.size, 0)
            if source.eligible:
                self.assertEqual(source.train_oof_score.shape, (len(self.values.train_indices),))
                self.assertEqual(source.test_pvalue.shape, source.test_score.shape)
                self.assertEqual(source.test_native_pass.shape, source.test_score.shape)
                np.testing.assert_allclose(source.test_score, second.by_name[name].test_score)
            else:
                self.assertEqual(source.train_oof_score.size, 0)
                self.assertEqual(source.test_pvalue.shape, (len(self.values.test_indices),))
                self.assertEqual(source.test_native_pass.shape, source.test_pvalue.shape)
                self.assertTrue(np.all(source.test_pvalue == 1.0))
                self.assertFalse(np.any(source.test_native_pass))

    def test_small_train_and_nonfinite_input_abstain_without_detector_substitution(self) -> None:
        """An ineligible raw input must abstain rather than use a replacement detector."""
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import score_hard_sources

        too_small = score_hard_sources(_raw_fixture(train_count=4))
        self.assertTrue(all(not item.eligible for item in too_small.by_name.values()))
        self.assertTrue(all(item.eligibility_reason == "unsupported_n_train_lt_5"
                            for item in too_small.by_name.values()))

        values = _raw_fixture()
        bad_train = list(values.train_rows)
        bad_train[0] = bad_train[0].copy()
        bad_train[0][0] = np.nan
        nonfinite = score_hard_sources(RawDataset(
            dataset=values.dataset,
            metadata_length=values.metadata_length,
            train_indices=values.train_indices, train_blobs=values.train_blobs,
            train_rows=tuple(bad_train), test_indices=values.test_indices,
            test_blobs=values.test_blobs, test_rows=values.test_rows,
        ))
        self.assertTrue(all(item.eligibility_reason == "nonfinite_input"
                            for item in nonfinite.by_name.values()))

    def test_source_metadata_cannot_derive_a_model_parameter_from_test_lengths(self) -> None:
        """Using TEST row lengths for the frozen target length would leak a model setting."""
        from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import source_metadata_paper

        values = _raw_fixture()
        changed = RawDataset(
            dataset=values.dataset, metadata_length=values.metadata_length,
            train_indices=values.train_indices, train_blobs=values.train_blobs,
            train_rows=values.train_rows, test_indices=values.test_indices,
            test_blobs=values.test_blobs,
            test_rows=tuple(np.concatenate((row, row, row)) for row in values.test_rows),
        )
        first, second = source_metadata_paper(values), source_metadata_paper(changed)
        self.assertEqual(first.train_target_len, second.train_target_len)
        self.assertEqual(first.test_actual_len_median, first.train_target_len)

    def test_paper_source_runtime_graph_has_no_legacy_or_label_capability(self) -> None:
        """A legacy runtime import or label dependency violates standalone score execution."""
        module = importlib.import_module(
            "paper.equipment_anomaly_detection.sources_paper.hard_sources_paper"
        )
        root = module.__file__
        self.assertIsNotNone(root)
        tree = ast.parse(Path(root).read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(any(name.startswith("experiments.exp") for name in imported))
        self.assertFalse(any("label_access_paper" in name for name in imported))

        source_root = Path(root).parent
        for source in source_root.glob("*_paper.py"):
            parsed = ast.parse(source.read_text(encoding="utf-8"))
            names = [node.module for node in ast.walk(parsed)
                     if isinstance(node, ast.ImportFrom) and node.module]
            self.assertFalse(any(name.startswith("experiments.exp") for name in names), source.name)

    def test_rocket_primitive_matches_isolated_frozen_origin(self) -> None:
        """Changing frozen ROCKET kernels or transform semantics must break score parity."""
        from paper.equipment_anomaly_detection.sources_paper.rocket_paper import (
            make_kernels_paper, rocket_transform_paper,
        )

        matrix = np.arange(64, dtype=np.float32).reshape(2, 32) / 10.0
        expected = rocket_transform_paper(matrix, make_kernels_paper(32, 4))
        code = """
import json, logging, sys, types
logging.FileHandler = lambda *args, **kwargs: logging.StreamHandler()
rank = types.ModuleType('run_rank_ensemble_calibration'); rank.DB_PATH=''; rank.load_dataset_data=lambda *x: None; rank.z_normalize=lambda x: x
threshold = types.ModuleType('run_rank_threshold_calibration'); threshold.STRATEGIES={}; threshold.predict_by_strategy=lambda *x: None; threshold.top_k_predictions=lambda *x: None
sys.modules['run_rank_ensemble_calibration']=rank; sys.modules['run_rank_threshold_calibration']=threshold
import numpy as np
import run_experiment_26_rocket as origin
x=np.arange(64,dtype=np.float32).reshape(2,32)/10.0
print(json.dumps(origin.rocket_transform(x, origin.make_kernels(32,4)).tolist()))
"""
        completed = subprocess.run([sys.executable, "-c", code], cwd=Path.cwd(),
                                   check=True, capture_output=True, text=True, timeout=30)
        np.testing.assert_allclose(expected, np.asarray(json.loads(completed.stdout)))


if __name__ == "__main__":
    unittest.main()
