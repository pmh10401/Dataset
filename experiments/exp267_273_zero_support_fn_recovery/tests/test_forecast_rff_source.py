from __future__ import annotations

import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery.config import (
    RFF_MAX_TRAIN_RESIDUAL_VALUES,
    SEED,
)
from experiments.exp267_273_zero_support_fn_recovery import (
    forecast_rff_source as rff_module,
)
from experiments.exp267_273_zero_support_fn_recovery.forecast_rff_source import (
    RIDGE_ALPHAS,
    RFFNormalForecastResidual,
    estimate_train_residual_values,
    exact_spooled_quantile,
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

    def test_fold_group_statistics_are_reused_across_nested_alpha_checks(self) -> None:
        result = RFFNormalForecastResidual().score(bundle(n_train=20), SEED, self.path)

        self.assertEqual(result.provenance["global_fold_count"], 5)
        self.assertEqual(result.provenance["fold_group_statistics_builds"], 5)
        self.assertEqual(
            result.provenance["fold_group_context_statistics_builds"], 15
        )
        self.assertEqual(
            result.provenance["statistics_train_row_transforms"], 20 * 3
        )
        self.assertEqual(result.provenance["alpha_candidate_count"], 4)
        self.assertEqual(result.provenance["ridge_eigendecompositions"], 16 * 3)
        self.assertEqual(result.provenance["all_alpha_validation_calls"], 25)

    def test_exact_spooled_quantile_matches_numpy_linear_interpolation(self) -> None:
        values = np.asarray([9.0, 2.0, 7.0, 1.0, 4.0, 12.0], dtype=float)

        actual = exact_spooled_quantile(
            (values[:2], values[2:]),
            value_count=len(values),
            artifact_dir=self.path,
            quantile=0.35,
        )

        self.assertEqual(actual, np.quantile(values, 0.35, method="linear"))
        self.assertFalse(list(self.path.glob("exp270_rff_residual_*.dat")))

    def test_spooled_quantile_cleans_temporary_file_after_generator_failure(self) -> None:
        def broken_chunks():
            yield np.asarray([1.0, 2.0], dtype=float)
            raise RuntimeError("intentional spool failure")

        with self.assertRaisesRegex(RuntimeError, "intentional spool failure"):
            exact_spooled_quantile(
                broken_chunks(),
                value_count=3,
                artifact_dir=self.path,
                quantile=0.95,
            )
        self.assertFalse(list(self.path.glob("exp270_rff_residual_*.dat")))

    def test_large_train_residual_estimate_abstains_before_rff_work(self) -> None:
        target_length = 250_000
        train = np.zeros((20, target_length), dtype=np.float32)
        oversized = DatasetBundle(
            dataset="oversized",
            train_normal=train,
            test_values=train[:1],
            target_length=target_length,
        )
        self.assertGreater(
            estimate_train_residual_values(20, target_length),
            RFF_MAX_TRAIN_RESIDUAL_VALUES,
        )

        with patch.object(rff_module, "_build_fold_group_statistics") as build:
            result = RFFNormalForecastResidual().score(oversized, SEED, self.path)

        build.assert_not_called()
        self.assertTrue(result.abstained)
        self.assertEqual(
            result.abstention_reason, "train_residual_work_exceeds_10000000"
        )
        self.assertEqual(
            result.provenance["rff_max_train_residual_values"], 10_000_000
        )

    def test_standardization_references_use_selected_outer_and_full_train_models(self) -> None:
        captured = []
        original = rff_module._robust_standardize_features

        def observe(reference, values):
            captured.append((reference.shape[0], values.shape[0]))
            return original(reference, values)

        with patch.object(rff_module, "_robust_standardize_features", side_effect=observe):
            result = RFFNormalForecastResidual().score(bundle(n_train=20), SEED, self.path)

        self.assertEqual(captured[:-1], [(16, 4)] * 5)
        self.assertEqual(captured[-1], (20, 2))
        self.assertEqual(
            result.provenance["oof_standardization_reference"],
            "selected_outer_model_on_outer_fit_rows",
        )
        self.assertEqual(
            result.provenance["test_standardization_reference"],
            "full_model_on_all_train_rows",
        )
        self.assertEqual(result.provenance["test_reference_row_count"], 20)

    def test_end_to_end_smoke_records_bounded_spool_work(self) -> None:
        smoke = bundle(n_train=40, target_length=128)
        tracemalloc.start()
        result = RFFNormalForecastResidual().score(smoke, SEED, self.path)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertFalse(result.abstained)
        self.assertGreater(peak_bytes, 0)
        self.assertLessEqual(
            result.provenance["estimated_train_residual_values"],
            RFF_MAX_TRAIN_RESIDUAL_VALUES,
        )
        self.assertLessEqual(
            result.provenance["max_residual_spool_values"],
            RFF_MAX_TRAIN_RESIDUAL_VALUES,
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
