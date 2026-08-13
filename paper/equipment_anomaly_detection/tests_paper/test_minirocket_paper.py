"""Standalone MiniROCKET closure contracts, without legacy runtime imports."""
from __future__ import annotations

import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
    repair_linear_edge_nearest_antigravity,
)
from paper.equipment_anomaly_detection.sources_paper import minirocket_paper
from paper.equipment_anomaly_detection.sources_paper import preprocessing_paper
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    ConfigurationScoresPaper,
    DependencyUnavailablePaper,
    MiniRocketConfig,
    MiniRocketNonfiniteInputPaper,
    expected_feature_dimension_paper,
    score_configuration_paper,
    score_minirocket_confirmations,
)
from paper.equipment_anomaly_detection.contracts_paper import RawDataset


class _Transformer:
    def fit_transform(self, rows: np.ndarray) -> np.ndarray:
        return rows[:, 0, :2].astype(float)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        return rows[:, 0, :2].astype(float)


class MiniRocketTests(unittest.TestCase):
    def test_linear_edge_nearest_repair_is_byte_identical_to_exp443_policy_a(self) -> None:
        fixtures = (
            np.asarray([1.0, np.nan, 3.0], dtype="<f4"),
            np.asarray([np.nan, 2.0, 4.0, np.nan], dtype="<f4"),
            np.asarray([1.0, np.inf, -np.inf, 4.0], dtype="<f4"),
            np.asarray([np.nan, 7.0, np.nan], dtype="<f4"),
            np.asarray([1.0, 2.0, 3.0], dtype="<f4"),
        )
        repair = preprocessing_paper.repair_linear_edge_nearest_paper
        for values in fixtures:
            actual = repair(values)
            expected = repair_linear_edge_nearest_antigravity(values)
            self.assertEqual(actual.dtype, np.dtype("<f4"))
            self.assertEqual(actual.tobytes(), expected.tobytes())
        with self.assertRaisesRegex(ValueError, "entirely non-finite"):
            repair(np.asarray([np.nan, np.inf], dtype="<f4"))

    def test_confirmations_repair_train_and_test_before_alignment(self) -> None:
        values = RawDataset(
            "nonfinite_fixture", 6, tuple(range(5)), (b"a",) * 5,
            tuple(np.asarray([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], dtype="<f4") for _ in range(5)),
            tuple(range(2)), (b"b",) * 2,
            tuple(np.asarray([np.nan, 2.0, 3.0, 4.0, 5.0, np.inf], dtype="<f4") for _ in range(2)),
        )
        observed: list[tuple[np.ndarray, np.ndarray]] = []

        def fake_score(train, test, *, requested_kernels, **_):
            observed.append((np.array(train, copy=True), np.array(test, copy=True)))
            return ConfigurationScoresPaper(
                np.arange(len(train), dtype=float), np.arange(len(test), dtype=float),
                1008 if requested_kernels == 1024 else 4032,
            )

        with patch.object(minirocket_paper, "score_configuration_paper", side_effect=fake_score):
            score_minirocket_confirmations(values)
        self.assertEqual(len(observed), 4)
        self.assertTrue(all(np.isfinite(matrix).all() for pair in observed for matrix in pair))

    def test_confirmations_report_dataset_local_context_for_unrepairable_rows(self) -> None:
        values = RawDataset(
            "invalid_fixture", 6, tuple(range(5)), (b"a",) * 5,
            tuple(np.arange(6, dtype="<f4") for _ in range(5)),
            tuple(range(2)), (b"b",) * 2,
            (np.arange(6, dtype="<f4"), np.full(6, np.nan, dtype="<f4")),
        )
        with self.assertRaisesRegex(
            MiniRocketNonfiniteInputPaper,
            r"dataset=invalid_fixture split=TEST row=1: series is entirely non-finite",
        ):
            score_minirocket_confirmations(values)

    def test_small_train_short_circuits_before_nonfinite_repair(self) -> None:
        values = RawDataset(
            "small_fixture", 6, tuple(range(4)), (b"a",) * 4,
            tuple(np.full(6, np.nan, dtype="<f4") for _ in range(4)),
            (0,), (b"b",), (np.full(6, np.nan, dtype="<f4"),),
        )
        with patch.object(minirocket_paper, "repair_linear_edge_nearest_paper") as repair:
            result = score_minirocket_confirmations(values)
        repair.assert_not_called()
        self.assertEqual(result.k1024, frozenset())
        self.assertEqual(result.k4096, frozenset())

    def test_requested_kernel_feature_dimensions_are_frozen(self) -> None:
        config = MiniRocketConfig()
        self.assertEqual(expected_feature_dimension_paper(config, 1024), 1008)
        self.assertEqual(expected_feature_dimension_paper(config, 4096), 4032)

    def test_crossfit_scores_are_finite_with_a_local_factory(self) -> None:
        train = np.arange(60, dtype=np.float32).reshape(10, 6)
        test = np.arange(18, dtype=np.float32).reshape(3, 6)
        result = score_configuration_paper(
            train, test, config=MiniRocketConfig(), requested_kernels=1024,
            seed=20260721, expected_dimension=None, transformer_factory=lambda **_: _Transformer(),
        )
        self.assertEqual(result.train_oof.shape, (10,))
        self.assertEqual(result.test_score.shape, (3,))
        self.assertTrue(np.isfinite(result.train_oof).all())
        self.assertTrue(np.isfinite(result.test_score).all())

    def test_optional_aeon_absence_is_typed_but_other_errors_are_not_masked(self) -> None:
        values = RawDataset("toy", 6, tuple(range(5)), (b"a",) * 5, tuple(np.ones(6, dtype=np.float32) for _ in range(5)), tuple(range(2)), (b"b",) * 2, tuple(np.ones(6, dtype=np.float32) for _ in range(2)))
        try:
            result = score_minirocket_confirmations(values)
        except DependencyUnavailablePaper:
            return
        self.assertEqual(set(result.by_kernel), {1024, 4096})

    def test_capable_oracle_matches_both_kernels_and_seeds(self) -> None:
        """Feature dimensions, OOF/TEST scores, p-values, and candidates stay exact."""
        oracle = Path("/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-py314/bin/python")
        if not oracle.is_file():
            self.skipTest("capable frozen oracle interpreter is unavailable")
        code = r'''
import numpy as np, sys
import aeon, sklearn
assert sys.version_info[:3] == (3, 14, 6)
assert aeon.__version__ == '1.5.0' and sklearn.__version__ == '1.8.0'
from experiments.exp249_250_official_mini_multirocket.config import MODEL_CONFIGS
from experiments.exp249_250_official_mini_multirocket.scoring import score_configuration
from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import MiniRocketConfig, score_configuration_paper
rng=np.random.default_rng(20260812); train=rng.normal(size=(10,16)).astype(np.float32); test=rng.normal(size=(3,16)).astype(np.float32)
for kernels, dimension in ((1024,1008),(4096,4032)):
  for seed in (20260721,20261721):
    origin=score_configuration(train,test,config=MODEL_CONFIGS[249],requested_kernels=kernels,seed=seed,expected_dimension=dimension)
    paper=score_configuration_paper(train,test,config=MiniRocketConfig(),requested_kernels=kernels,seed=seed,expected_dimension=dimension)
    assert paper.feature_dimension == origin.feature_dimension == dimension
    np.testing.assert_allclose(paper.train_oof,origin.train_oof,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(paper.test_score,origin.test_scores,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(upper_tail_pvalues(paper.train_oof,paper.test_score),upper_tail_pvalues(origin.train_oof,origin.test_scores))
    assert np.array_equal(upper_tail_pvalues(paper.train_oof,paper.test_score)<=.005, upper_tail_pvalues(origin.train_oof,origin.test_scores)<=.005)
print('minirocket-parity')
'''
        completed = subprocess.run([str(oracle), "-c", code], cwd=Path.cwd(), check=True, capture_output=True, text=True, timeout=180)
        self.assertIn("minirocket-parity", completed.stdout)


if __name__ == "__main__":
    unittest.main()
