import unittest
from types import SimpleNamespace
import numpy as np

from services.time_series_anomaly_diagnosis.anomaly_detection import (
    DetectionResult,
    FIXED_SEEDS,
    calculate_evaluation_metrics,
    empirical_upper_pvalues,
    run_anomaly_detection,
    run_analysis_policy,
    run_method_detection,
    stable_descending_indices,
    _empirical_upper_pvalues_numba,
)
from services.time_series_anomaly_diagnosis.score_sources import ScoreResult
from services.time_series_anomaly_diagnosis.dataset_repository import PreparedDatasetValues
from services.time_series_anomaly_diagnosis.routing_policy import (
    FEATURE_KEYS,
    MethodMetric,
    RouterTrainingRecord,
    fit_train_feature_router,
)


def router_record(group_id, value):
    return RouterTrainingRecord(
        group_id,
        dict.fromkeys(FEATURE_KEYS, value),
        tuple(MethodMetric(name, 0.8, int(name != "minirocket_local_gap")) for name in (
            "minirocket_local_gap", "multirocket_local_gap", "hydra_local_gap"
        )),
    )


class AnomalyDetectionTest(unittest.TestCase):
    def policy_prepared(self, n_train=199, raw=True):
        return PreparedDatasetValues(
            "name_is_not_a_feature", np.zeros((n_train, 2)), np.zeros((4, 2)),
            (8, 49, 51, 60), "standard",
            tuple(np.array([i, i + 1.0]) for i in range(max(2, n_train))) if raw else None,
        )

    def policy_scorer(self, calls, final_scores):
        train = np.arange(199, dtype=float)
        def scorer(_train, _test, method, seed):
            calls.append((method, seed))
            return ScoreResult(train, np.asarray(final_scores[method][seed], dtype=float), 2, seed)
        return scorer

    def test_policy_ids_calls_and_router_requirements(self):
        scores = {name: {seed: [300, 300, 300, 300] for seed in FIXED_SEEDS}
                  for name in ("minirocket_local_gap", "multirocket_local_gap", "hydra_local_gap")}
        prepared = self.policy_prepared()
        calls = []
        a = run_analysis_policy(prepared, "single_minirocket", scorer=self.policy_scorer(calls, scores))
        self.assertEqual([method for method, _ in calls], ["minirocket_local_gap"] * 2)
        self.assertEqual(a.policy_id, "single_minirocket")
        calls.clear()
        b = run_analysis_policy(prepared, "three_method_consensus", scorer=self.policy_scorer(calls, scores))
        self.assertEqual([method for method, _ in calls], [name for name in scores for _ in FIXED_SEEDS])
        self.assertEqual(b.method_name, "three_method_consensus")
        self.assertIsNone(b.selected_method)
        router = fit_train_feature_router([router_record(i, float(i)) for i in range(20)])
        calls.clear()
        c = run_analysis_policy(prepared, "train_feature_router", router, scorer=self.policy_scorer(calls, scores))
        self.assertEqual(len(calls), 2)
        self.assertEqual(c.selected_method, "multirocket_local_gap")
        with self.assertRaises(ValueError):
            run_analysis_policy(prepared, "unknown", scorer=self.policy_scorer([], scores))
        with self.assertRaisesRegex(ValueError, "fitted router"):
            run_analysis_policy(prepared, "train_feature_router", scorer=self.policy_scorer([], scores))
        with self.assertRaisesRegex(ValueError, "fitted router"):
            run_analysis_policy(prepared, "train_feature_router", SimpleNamespace(route=lambda _: None),
                                scorer=self.policy_scorer([], scores))
        with self.assertRaisesRegex(ValueError, "raw TRAIN"):
            run_analysis_policy(self.policy_prepared(raw=False), "train_feature_router", router,
                                scorer=self.policy_scorer([], scores))
        run_analysis_policy(self.policy_prepared(raw=False), "single_minirocket",
                            scorer=self.policy_scorer([], scores))
        run_analysis_policy(self.policy_prepared(raw=False), "three_method_consensus",
                            scorer=self.policy_scorer([], scores))
        renamed = PreparedDatasetValues(
            "renamed", prepared.train_values, prepared.test_values, prepared.test_indices,
            prepared.preparation_mode, prepared.raw_train_rows,
        )
        c_renamed = run_analysis_policy(renamed, "train_feature_router", router,
                                        scorer=self.policy_scorer([], scores))
        self.assertEqual(c.selected_method, c_renamed.selected_method)

    def test_policy_automatic_review_overlap_exclusion_and_scores(self):
        floor = 1 / 200
        train = np.arange(199, dtype=float)
        # TEST score > 198 gives the attainable minimum 0.005.
        values = {
            "minirocket_local_gap": {20260721: [300, 300, 300, 0], 20261721: [300, 300, 300, 0]},
            "multirocket_local_gap": {20260721: [300, 300, 0, 0], 20261721: [300, 300, 0, 0]},
            "hydra_local_gap": {20260721: [0, 0, 300, 0], 20261721: [0, 0, 300, 0]},
        }
        prepared = self.policy_prepared()
        result = run_analysis_policy(prepared, "three_method_consensus",
                                     scorer=self.policy_scorer([], values))
        self.assertEqual(result.automatic_anomaly_indices, (8, 49, 51))
        self.assertEqual(result.seed_consensus_review_indices, ())
        self.assertEqual(result.method_consensus_review_indices, ())
        self.assertTrue(set(result.automatic_anomaly_indices).isdisjoint(result.review_indices))
        self.assertEqual(result.rows[0]["p_value"], floor)
        self.assertEqual(result.rows[0]["score"], 1 - floor)
        self.assertEqual(set(result.rows[0]), {
            "series_index", "score", "p_value", "automatic_anomaly",
            "method_p_values", "method_seed_p_values", "supporting_methods",
        })

    def test_policy_review_boundaries_overlap_and_no_promotion(self):
        prepared = self.policy_prepared(n_train=40)
        train = np.arange(40, dtype=float)
        values = {
            "minirocket_local_gap": {seed: [100, 100, 100, 0] for seed in FIXED_SEEDS},
            "multirocket_local_gap": {seed: [0, 100, 100, 0] for seed in FIXED_SEEDS},
            "hydra_local_gap": {seed: [0, 0, 100, 0] for seed in FIXED_SEEDS},
        }
        def scorer(_train, _test, method, seed):
            return ScoreResult(train, np.array(values[method][seed]), 2, seed)
        b = run_analysis_policy(prepared, "three_method_consensus", scorer=scorer)
        self.assertEqual(b.automatic_anomaly_indices, ())
        self.assertEqual(b.seed_consensus_review_indices, (49, 51, 8))
        self.assertEqual(b.method_consensus_review_indices, (49, 51))
        self.assertEqual(b.review_indices, (49, 51, 8))
        self.assertFalse(any(row["automatic_anomaly"] for row in b.rows))
        a = run_analysis_policy(prepared, "single_minirocket", scorer=scorer)
        self.assertEqual(a.seed_consensus_review_indices, (8, 49, 51))
        self.assertEqual(a.method_consensus_review_indices, ())

    def test_policy_threshold_equality_stable_p_sort_and_alignment_validation(self):
        prepared = PreparedDatasetValues("x", np.zeros((199, 2)), np.zeros((3, 2)),
                                         (30, 10, 20), "standard")
        train = np.arange(199, dtype=float)
        def scorer(_train, _test, _method, seed):
            return ScoreResult(train, np.array([300.0, 300.0, 0.0]), 2, seed)
        result = run_analysis_policy(prepared, "single_minirocket", scorer=scorer)
        self.assertEqual([row["series_index"] for row in result.rows], [30, 10, 20])
        self.assertEqual(result.automatic_anomaly_indices, (30, 10))
        def misaligned(_train, _test, _method, seed):
            return ScoreResult(train, np.array([1.0]), 2, seed)
        with self.assertRaisesRegex(ValueError, "aligned"):
            run_analysis_policy(prepared, "single_minirocket", scorer=misaligned)

    def test_empirical_upper_pvalues_and_stable_tie_order(self) -> None:
        self.assertTrue(hasattr(_empirical_upper_pvalues_numba, "py_func"))
        np.testing.assert_allclose(
            empirical_upper_pvalues(
                np.array([0.0, 1.0, 2.0]),
                np.array([2.0, 3.0]),
            ),
            [0.5, 0.25],
        )
        self.assertEqual(stable_descending_indices(np.array([4.0, 4.0])), (0, 1))

    def test_standard_detection_uses_two_seed_median_original_indices_and_stable_sort(self) -> None:
        prepared = PreparedDatasetValues(
            "sample",
            np.zeros((199, 4)),
            np.zeros((4, 4)),
            (10, 20, 30, 40),
            "standard",
        )
        called_seeds = []
        train = {
            20260721: np.arange(199, dtype=float),
            20261721: np.arange(199, dtype=float) + 2.0,
        }
        # position 1: median(300, 100) = 200.0 (high -> anomaly)
        # position 0: median(50, 50) = 50.0
        # position 3: median(50, 50) = 50.0 (tie with position 0, stable sort preserves 0 before 3)
        # position 2: median(0, 2) = 1.0
        test = {
            20260721: np.array([50.0, 300.0, 0.0, 50.0]),
            20261721: np.array([50.0, 100.0, 2.0, 50.0]),
        }

        def scorer(_train, _test, method_name, seed):
            called_seeds.append(seed)
            self.assertEqual(method_name, "minirocket_local_gap")
            return ScoreResult(train[seed], test[seed], 2, seed)

        result = run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer)
        self.assertIsInstance(result, DetectionResult)
        self.assertEqual(called_seeds, [20260721, 20261721])

        # test_indices: (10, 20, 30, 40)
        # sorted order of positions by -test_scores:
        # position 1 (score 200.0) -> index 20
        # position 0 (score 50.0) -> index 10 (mergesort tie-breaker keeps position 0 before 3)
        # position 3 (score 50.0) -> index 40
        # position 2 (score 1.0) -> index 30
        self.assertEqual([row["series_index"] for row in result.rows], [20, 10, 40, 30])
        self.assertEqual(result.automatic_anomaly_indices, (20,))

        # Verify row dict schema strictly contains only 4 keys
        for row in result.rows:
            self.assertEqual(
                set(row.keys()),
                {"series_index", "score", "p_value", "automatic_anomaly"},
            )
        self.assertEqual(result.rows[0]["score"], 200.0)
        self.assertTrue(result.rows[0]["automatic_anomaly"])
        self.assertFalse(result.rows[1]["automatic_anomaly"])

    def test_recovery_detection_uses_same_two_seeds_as_standard(self) -> None:
        prepared = PreparedDatasetValues(
            "sample_recovery",
            np.zeros((199, 3)),
            np.zeros((2, 3)),
            (100, 200),
            "recovery_numba",
        )
        called_seeds = []

        def scorer(_train, _test, _method_name, seed):
            called_seeds.append(seed)
            return ScoreResult(
                np.arange(199, dtype=float), np.array([10.0, 20.0]), 2, seed
            )

        result = run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer)
        self.assertEqual(FIXED_SEEDS, (20260721, 20261721))
        self.assertEqual(called_seeds, [20260721, 20261721])
        self.assertEqual([row["series_index"] for row in result.rows], [200, 100])

        unsupported = PreparedDatasetValues(
            "sample_unavailable",
            prepared.train_values,
            prepared.test_values,
            prepared.test_indices,
            "unavailable",
        )
        with self.assertRaises(ValueError):
            run_anomaly_detection(unsupported, "minirocket_local_gap", scorer=scorer)

    def test_unattainable_alpha_promotes_floor_candidate_with_warning(self) -> None:
        prepared = PreparedDatasetValues(
            "small_sample",
            np.zeros((40, 3)),
            np.zeros((2, 3)),
            (49, 7),
            "standard",
        )

        def scorer(_train, _test, _method_name, seed):
            return ScoreResult(
                np.arange(40, dtype=float), np.array([100.0, 39.0]), 2, seed
            )

        result = run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer)

        self.assertEqual(result.automatic_anomaly_indices, (49,))
        self.assertEqual(result.resolution_limited_warning_indices, (49,))
        row_49 = next(row for row in result.rows if row["series_index"] == 49)
        self.assertTrue(row_49["automatic_anomaly"])

    def test_train_reference_and_target_length(self) -> None:
        train_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ])
        test_matrix = np.zeros((2, 3))
        prepared = PreparedDatasetValues(
            "sample",
            train_matrix,
            test_matrix,
            (0, 1),
            "standard",
        )

        def scorer(_train, _test, _method_name, seed):
            return ScoreResult(
                np.arange(5, dtype=float), np.array([1.0, 2.0]), 2, seed
            )

        result = run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer)
        self.assertEqual(result.target_length, 3)

        expected_median = np.median(train_matrix, axis=0).astype(float).tolist()
        expected_q10 = np.quantile(train_matrix, 0.10, axis=0).astype(float).tolist()
        expected_q90 = np.quantile(train_matrix, 0.90, axis=0).astype(float).tolist()

        self.assertEqual(result.train_reference["median"], expected_median)
        self.assertEqual(result.train_reference["q10"], expected_q10)
        self.assertEqual(result.train_reference["q90"], expected_q90)
        for key in ("median", "q10", "q90"):
            self.assertTrue(all(isinstance(v, float) for v in result.train_reference[key]))

    def test_evaluation_metrics_success_example(self) -> None:
        rows = (
            {"series_index": 0, "score": 0.1, "p_value": 0.5, "automatic_anomaly": False},
            {"series_index": 1, "score": 0.9, "p_value": 0.001, "automatic_anomaly": True},
            {"series_index": 2, "score": 0.8, "p_value": 0.002, "automatic_anomaly": True},
            {"series_index": 3, "score": 0.2, "p_value": 0.3, "automatic_anomaly": False},
        )
        labels = {0: 0, 1: 1, 2: 0, 3: 1}
        metrics = calculate_evaluation_metrics(rows, labels)
        self.assertEqual(
            set(metrics.keys()),
            {"auc", "true_positives", "false_positives", "true_negatives", "false_negatives"},
        )
        self.assertAlmostEqual(metrics["auc"], 0.75)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_evaluation_metrics_raises_on_missing_label_or_single_class(self) -> None:
        rows = (
            {"series_index": 0, "score": 0.1, "automatic_anomaly": False},
            {"series_index": 1, "score": 0.9, "automatic_anomaly": True},
        )
        # Missing label for series_index 1
        with self.assertRaises(ValueError):
            calculate_evaluation_metrics(rows, {0: 0})

        # Single class (only 0)
        with self.assertRaises(ValueError):
            calculate_evaluation_metrics(rows, {0: 0, 1: 0})

        # Single class (only 1)
        with self.assertRaises(ValueError):
            calculate_evaluation_metrics(rows, {0: 1, 1: 1})

        with self.assertRaises(ValueError):
            calculate_evaluation_metrics((), {0: 0, 1: 1})

        metrics = calculate_evaluation_metrics((row for row in rows), {0: 0, 1: 1})
        self.assertEqual(metrics["true_positives"], 1)

    def test_non_finite_scores_raise_value_error(self) -> None:
        prepared = PreparedDatasetValues(
            "sample",
            np.zeros((5, 2)),
            np.zeros((2, 2)),
            (0, 1),
            "standard",
        )

        def scorer_nan_train(_train, _test, _method_name, seed):
            return SimpleNamespace(
                train_oof=np.array([1.0, np.nan, 2.0, 3.0, 4.0]),
                test_scores=np.array([1.0, 2.0]),
            )

        with self.assertRaises(ValueError):
            run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer_nan_train)

        def scorer_inf_test(_train, _test, _method_name, seed):
            return SimpleNamespace(
                train_oof=np.arange(5, dtype=float),
                test_scores=np.array([1.0, np.inf]),
            )

        with self.assertRaises(ValueError):
            run_anomaly_detection(prepared, "minirocket_local_gap", scorer=scorer_inf_test)

        rows = (
            {"series_index": 0, "score": np.nan, "automatic_anomaly": False},
            {"series_index": 1, "score": 0.9, "automatic_anomaly": True},
        )
        with self.assertRaises(ValueError):
            calculate_evaluation_metrics(rows, {0: 0, 1: 1})

    def test_method_detection_recomputes_final_pvalues_from_median_scores(self) -> None:
        prepared = PreparedDatasetValues(
            "sample",
            np.zeros((5, 2)),
            np.zeros((2, 2)),
            (0, 1),
            "standard",
        )
        results = {
            20260721: ScoreResult(
                np.array([0.0, 0.0, 0.0, 0.0, 100.0]),
                np.array([40.0, 70.0]),
                3,
                20260721,
            ),
            20261721: ScoreResult(
                np.array([60.0, 60.0, 60.0, 60.0, 60.0]),
                np.array([80.0, 30.0]),
                3,
                20261721,
            ),
        }

        detection = run_method_detection(
            prepared,
            "minirocket_local_gap",
            scorer=lambda _train, _test, _method, seed: results[seed],
        )

        np.testing.assert_array_equal(detection.train_scores, [30.0, 30.0, 30.0, 30.0, 80.0])
        np.testing.assert_array_equal(detection.test_scores, [60.0, 50.0])
        np.testing.assert_allclose(detection.final_pvalues, [2 / 6, 2 / 6])
        np.testing.assert_allclose(detection.seed_pvalues[0], [2 / 6, 2 / 6])
        np.testing.assert_allclose(detection.seed_pvalues[1], [1 / 6, 1.0])
        self.assertNotEqual(
            detection.final_pvalues[1],
            np.median([detection.seed_pvalues[0][1], detection.seed_pvalues[1][1]]),
        )


if __name__ == "__main__":
    unittest.main()
