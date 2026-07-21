from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery.config import SEED
from experiments.exp267_273_zero_support_fn_recovery.forecast_rff_source import (
    RIDGE_ALPHAS,
    RFFNormalForecastResidual,
    forecast_context_lengths,
    forecast_horizon,
    select_ridge_alpha,
)


def bundle(*, n_train: int = 20, target_length: int = 64) -> DatasetBundle:
    generator = np.random.default_rng(270)
    time = np.linspace(0.0, 6.0 * np.pi, target_length)
    train = np.vstack(
        [
            np.sin(time + offset)
            + 0.05 * np.sin(3.0 * time + offset)
            + generator.normal(0.0, 0.015, target_length)
            for offset in np.linspace(0.0, 0.3, n_train)
        ]
    )
    normal = np.sin(time) + 0.05 * np.sin(3.0 * time)
    unusual = normal.copy()
    unusual[target_length // 2 :] += 1.25
    return DatasetBundle(
        dataset="forecast-demo",
        train_normal=train.astype(np.float32),
        test_values=np.vstack((normal, unusual)).astype(np.float32),
        target_length=target_length,
    )


class RFFNormalForecastResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_rff_abstains_below_20_train_rows(self) -> None:
        result = RFFNormalForecastResidual().score(bundle(n_train=19), SEED, self.path)

        self.assertTrue(result.abstained)
        self.assertFalse(result.eligible)
        self.assertEqual(result.abstention_reason, "n_train_below_20")
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_oof_query_never_appears_in_fit_indices(self) -> None:
        result = RFFNormalForecastResidual().score(bundle(n_train=20), SEED, self.path)

        self.assertFalse(result.abstained)
        self.assertEqual(result.provenance["self_reference_violations"], 0)
        self.assertEqual(result.provenance["alpha_selection_query_fit_violations"], 0)
        self.assertTrue(result.provenance["cross_fitted"])
        self.assertEqual(len(result.provenance["outer_fit_query_overlap_counts"]), 5)
        self.assertTrue(
            all(count == 0 for count in result.provenance["outer_fit_query_overlap_counts"])
        )

    def test_scores_are_finite_deterministic_and_use_train_only_calibration(self) -> None:
        original = bundle(n_train=20)
        changed_test = DatasetBundle(
            dataset=original.dataset,
            train_normal=original.train_normal,
            test_values=np.flip(original.test_values, axis=0) * 7.0,
            target_length=original.target_length,
        )

        first = RFFNormalForecastResidual().score(original, SEED, self.path)
        second = RFFNormalForecastResidual().score(original, SEED, self.path)
        changed = RFFNormalForecastResidual().score(changed_test, SEED, self.path)

        np.testing.assert_array_equal(first.train_oof, second.train_oof)
        np.testing.assert_array_equal(first.test_scores, second.test_scores)
        np.testing.assert_array_equal(first.train_oof, changed.train_oof)
        self.assertEqual(first.provenance["selected_alpha_test"], changed.provenance["selected_alpha_test"])
        self.assertTrue(np.isfinite(first.train_oof).all())
        self.assertTrue(np.isfinite(first.test_scores).all())
        self.assertFalse(first.provenance["contains_test_labels"])
        self.assertNotIn("y_test", first.provenance)

    def test_trace_with_shift_has_larger_forecast_residual_score(self) -> None:
        result = RFFNormalForecastResidual().score(bundle(n_train=24), SEED, self.path)

        self.assertGreater(result.test_scores[1], result.test_scores[0])

    def test_context_and_horizon_are_clipped_from_train_target_length(self) -> None:
        self.assertEqual(forecast_horizon(2), 1)
        self.assertEqual(forecast_context_lengths(2), (1,))
        self.assertEqual(forecast_horizon(64), 4)
        self.assertEqual(forecast_context_lengths(64), (8, 16, 32))
        self.assertEqual(forecast_horizon(128), 8)
        self.assertEqual(forecast_context_lengths(128), (8, 16, 32))

    def test_configuration_derives_from_train_width_not_bundle_metadata(self) -> None:
        source_bundle = bundle(n_train=20, target_length=32)
        misleading_metadata = DatasetBundle(
            dataset=source_bundle.dataset,
            train_normal=source_bundle.train_normal,
            test_values=source_bundle.test_values,
            target_length=4,
        )

        result = RFFNormalForecastResidual().score(
            misleading_metadata, SEED, self.path
        )

        self.assertFalse(result.abstained)
        self.assertEqual(result.provenance["target_length"], 32)
        self.assertEqual(result.provenance["forecast_horizon"], 2)
        self.assertEqual(result.provenance["context_lengths"], (8, 16, 30))

    def test_one_point_trace_abstains_and_two_point_trace_is_finite(self) -> None:
        one_point = bundle(n_train=20, target_length=1)
        abstained = RFFNormalForecastResidual().score(one_point, SEED, self.path)

        self.assertTrue(abstained.abstained)
        self.assertEqual(abstained.abstention_reason, "target_length_below_2")

        two_point = RFFNormalForecastResidual().score(
            bundle(n_train=20, target_length=2), SEED, self.path
        )
        self.assertFalse(two_point.abstained)
        self.assertTrue(np.isfinite(two_point.train_oof).all())
        self.assertTrue(np.isfinite(two_point.test_scores).all())

    def test_constant_train_rows_remain_finite_and_numeric_alpha_tie_breaks(self) -> None:
        rows = np.zeros((20, 8), dtype=np.float32)
        constant = DatasetBundle(
            dataset="constant",
            train_normal=rows,
            test_values=rows[:2],
            target_length=8,
        )

        selected, losses = select_ridge_alpha(rows, SEED)
        result = RFFNormalForecastResidual().score(constant, SEED, self.path)

        self.assertEqual(RIDGE_ALPHAS, (0.01, 0.1, 1.0, 10.0))
        self.assertEqual(selected, 0.01)
        self.assertEqual(tuple(losses), RIDGE_ALPHAS)
        self.assertTrue(np.isfinite(result.train_oof).all())
        self.assertTrue(np.isfinite(result.test_scores).all())

    def test_test_length_mismatch_is_rejected_before_scoring(self) -> None:
        source_bundle = bundle(n_train=20, target_length=32)
        malformed = DatasetBundle(
            dataset=source_bundle.dataset,
            train_normal=source_bundle.train_normal,
            test_values=source_bundle.test_values[:, :-1],
            target_length=source_bundle.target_length,
        )

        with self.assertRaisesRegex(ValueError, "target length"):
            RFFNormalForecastResidual().score(malformed, SEED, self.path)


if __name__ == "__main__":
    unittest.main()
