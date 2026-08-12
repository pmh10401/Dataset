"""Focused behavioral contracts for the label-free hard score sources."""
from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

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

    def test_canonical_orchestration_passes_the_frozen_paper_seed(self) -> None:
        """A changed canonical fold seed changes OOF positions and native candidates."""
        from paper.equipment_anomaly_detection.sources_paper import hard_sources_paper

        with patch.object(hard_sources_paper, "canonical_features_paper", side_effect=lambda x: x), \
             patch.object(hard_sources_paper, "crossfit_scores_paper", return_value=(np.ones(5), np.ones(3))) as scored:
            hard_sources_paper._canonical_source_paper(np.ones((5, 8)), np.ones((3, 8)))
        self.assertEqual(scored.call_args.kwargs["seed"], hard_sources_paper.PAPER_SEED)

    def test_randomized_pca_branch_is_seeded_and_matches_sklearn(self) -> None:
        """Replacing the frozen large-matrix PCA branch must change the exact score source."""
        from sklearn.decomposition import PCA
        from paper.equipment_anomaly_detection.sources_paper.imaging_paper import _pca_pair_paper

        rng = np.random.default_rng(44)
        train = rng.normal(size=(1000, 1000)).astype(np.float32)
        test = rng.normal(size=(2, 1000)).astype(np.float32)
        actual_train, actual_test = _pca_pair_paper(train, test, 32)
        center = train.mean(axis=0, keepdims=True)
        expected = PCA(n_components=32, svd_solver="randomized", random_state=20260707)
        np.testing.assert_allclose(actual_train, expected.fit_transform(train - center), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(actual_test, expected.transform(test - center), rtol=1e-5, atol=1e-6)

    def test_source_provenance_binds_every_scientific_helper(self) -> None:
        """An unbound helper or copied origin would make scientific provenance mutable."""
        from paper.equipment_anomaly_detection.lineage_paper import validate_source_provenance_paper
        from paper.equipment_anomaly_detection.config_paper import PAPER_ROOT

        registry = validate_source_provenance_paper(PAPER_ROOT)
        self.assertIn("sources_paper/hard_sources_paper.py", registry)

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
            names = []
            for node in ast.walk(parsed):
                if isinstance(node, ast.Import):
                    names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
            self.assertFalse(any(name.startswith("experiments.exp") for name in names), source.name)

    def test_implementation_runtime_error_is_not_converted_to_abstention(self) -> None:
        """Only the typed missing-dependency signal may become a source abstention."""
        from paper.equipment_anomaly_detection.sources_paper import hard_sources_paper

        with patch.object(hard_sources_paper, "crossfit_confirmation_scores_paper", side_effect=RuntimeError("broken scorer")):
            with self.assertRaisesRegex(RuntimeError, "broken scorer"):
                hard_sources_paper.score_hard_sources(self.values)

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

    def test_capable_oracle_matches_every_source_scores_and_candidates(self) -> None:
        """Any source-level drift in crossfit, preprocessing, or native candidates fails parity."""
        oracle = Path("/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python")
        if not oracle.is_file():
            self.skipTest("capable frozen oracle interpreter is unavailable")
        code = r'''
import logging, numpy as np, sys
logging.FileHandler=lambda *args, **kwargs: logging.StreamHandler()
import aeon, pycatch22, sklearn
assert sys.version_info[:3] == (3, 14, 6)
assert aeon.__version__ == '1.5.0' and sklearn.__version__ == '1.8.0'
import run_experiment_149_150_virtual_run_conformal_policy as historical
import run_model_hard_research_experiments as model_hard
from experiments.exp153_157 import exp153_157_bridge as bridge
from experiments.exp177_185.core import canonical_features, crossfit_scores
from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import crossfit_confirmation_scores_paper, crossfit_source_paper, _canonical_source_paper
from paper.equipment_anomaly_detection.sources_paper.imaging_paper import EXP55_CONFIG_PAPER, imaging_score_pair_paper
count, length = 200, 32
grid=np.linspace(0, 2*np.pi, length, dtype=np.float32); rng=np.random.default_rng(20260812)
train=np.asarray([np.sin(grid+rng.normal(0,.03))+rng.normal(0,.01,length) for _ in range(count)], dtype=np.float32)
test=np.vstack([np.sign(np.sin(grid*5))*4, np.r_[np.zeros(length//2),np.ones(length//2)*20], np.cos(grid)]).astype(np.float32)
record={'actual_len_median':32, 'dataset_name':'__virtual_run_dataset__', 'family':'__family_neutral__'}
def pvalues_and_candidate(scores, query):
    p=upper_tail_pvalues(scores, query)
    return p, [np.flatnonzero(p <= alpha).tolist() for alpha in (.005,.01,.02,.05)]
for name in ('rocket_exp40','exp55_best','exp56_best','exp84'):
    origin=historical.crossfit_source(name, train, test, record)
    paper=crossfit_source_paper(name, train, test, 32)
    np.testing.assert_allclose(origin['train_oof_scores'], paper[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(origin['test_scores'], paper[1], rtol=1e-5, atol=1e-5)
    origin_p, origin_candidates=pvalues_and_candidate(origin['train_oof_scores'], origin['test_scores']); paper_p, paper_candidates=pvalues_and_candidate(paper[0], paper[1])
    np.testing.assert_allclose(origin_p, paper_p); assert origin_candidates == paper_candidates and origin_candidates[0]
origin_oof=np.full(len(train),np.nan)
for fit, query in bridge.exp152_core.fold_plan(len(train), seed=20260717):
    _, scores=bridge._block_b_score_pair(train[fit],train[query]); origin_oof[query]=scores
_, origin_test=bridge._block_b_score_pair(train,test); paper=crossfit_confirmation_scores_paper(train, test)
np.testing.assert_allclose(origin_oof, paper[0]); np.testing.assert_allclose(origin_test, paper[1])
origin_p, origin_candidates=pvalues_and_candidate(origin_oof, origin_test); paper_p, paper_candidates=pvalues_and_candidate(paper[0], paper[1])
np.testing.assert_allclose(origin_p, paper_p); assert origin_candidates == paper_candidates and origin_candidates[0]
origin=crossfit_scores(canonical_features(train), canonical_features(test), backend='knn_local_gap', seed=20260717); paper=_canonical_source_paper(train,test)
np.testing.assert_allclose(origin[0],paper[0]); np.testing.assert_allclose(origin[1],paper[1])
origin_p, origin_candidates=pvalues_and_candidate(origin[0],origin[1]); paper_p, paper_candidates=pvalues_and_candidate(paper[0],paper[1])
np.testing.assert_allclose(origin_p,paper_p); assert origin_candidates == paper_candidates and origin_candidates[0]
large_count=1250; large_rng=np.random.default_rng(20260813)
large=np.asarray([np.sin(grid+large_rng.normal(0,.03))+large_rng.normal(0,.01,length) for _ in range(large_count)],dtype=np.float32)
origin=historical.crossfit_source('exp55_best',large,test,record)
import paper.equipment_anomaly_detection.sources_paper.imaging_paper as imaging
real_pca, pca_rows = imaging.PCA, []
def traced_pca(*args, **kwargs):
    model=real_pca(*args, **kwargs); real_fit=model.fit_transform
    def fit_transform(values, *fit_args, **fit_kwargs): pca_rows.append(len(values)); return real_fit(values, *fit_args, **fit_kwargs)
    model.fit_transform=fit_transform
    return model
imaging.PCA=traced_pca
paper=crossfit_source_paper('exp55_best',large,test,length)
assert pca_rows == [1000,1000,1000,1000,1000,1250], pca_rows
np.testing.assert_allclose(origin['train_oof_scores'],paper[0],rtol=1e-5,atol=1e-5); np.testing.assert_allclose(origin['test_scores'],paper[1],rtol=1e-5,atol=1e-5)
origin_p, origin_candidates=pvalues_and_candidate(origin['train_oof_scores'],origin['test_scores']); paper_p, paper_candidates=pvalues_and_candidate(paper[0],paper[1])
np.testing.assert_allclose(origin_p,paper_p); assert origin_candidates == paper_candidates and origin_candidates[0]
print('source-parity: rocket_exp40 rocket_block_b exp55_small exp55_randomized_crossfit exp56 exp84 canonical')
'''
        completed = subprocess.run([str(oracle), "-c", code], cwd=Path.cwd(),
                                   check=True, timeout=90)
        self.assertEqual(completed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
