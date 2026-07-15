import unittest
import numpy as np

from experiments.exp206_210.core import (
    fold_plan,
    leave_one_out_pvalues,
    classify_risk,
    empirical_upper_pvalues,
    robust_z_pair,
    top2_mean,
    inverse_spearman_weights,
    fisher_stat,
    wilson_upper,
    validate_prediction_payload,
)
from experiments.exp206_210.alternative_oof import (
    fit_score_source,
    crossfit_alternative_sources,
)


class Exp206210CoreTests(unittest.TestCase):

    def test_fold_plan_covers_every_index_once(self):
        folds = fold_plan(101, 5, 20260716)
        heldout = np.concatenate([query for _, query in folds])
        self.assertEqual(sorted(heldout.tolist()), list(range(101)))
        self.assertTrue(all(not set(fit) & set(query) for fit, query in folds))

    def test_leave_one_out_excludes_query_score(self):
        p = leave_one_out_pvalues(np.asarray([1.0, 2.0, 100.0]))
        self.assertAlmostEqual(p[-1], 1.0 / 3.0)

    def test_risk_boundaries_are_frozen(self):
        self.assertEqual(classify_risk(0.001), "shadow_hard_diagnostic")
        self.assertEqual(classify_risk(0.005), "standard_review_diagnostic")
        self.assertEqual(classify_risk(0.0051), "research_sensitivity_only")

    def test_empirical_upper_pvalues(self):
        ref = [1.0, 2.0, 3.0]
        query = [2.5, 0.5, 4.0]
        pvals = empirical_upper_pvalues(ref, query)
        np.testing.assert_allclose(pvals, [0.5, 1.0, 0.25])

    def test_robust_z_pair(self):
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        query = np.array([0.0, 6.0])
        tr_z, q_z = robust_z_pair(train, query)
        self.assertEqual(len(tr_z), 5)
        self.assertEqual(len(q_z), 2)
        np.testing.assert_allclose(tr_z, (train - 3.0) / 1.4826)
        np.testing.assert_allclose(q_z, (query - 3.0) / 1.4826)

    def test_top2_mean(self):
        mat = np.array([
            [1.0, 4.0, 2.0],
            [10.0, 5.0, 8.0]
        ])
        res = top2_mean(mat)
        np.testing.assert_allclose(res, [3.0, 9.0])

    def test_inverse_spearman_weights(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        weights = inverse_spearman_weights(x)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_fisher_stat(self):
        pvals = np.array([0.05, 0.01])
        stat = fisher_stat(pvals)
        self.assertAlmostEqual(stat, -2.0 * (np.log(0.05) + np.log(0.01)))

    def test_wilson_upper(self):
        upper = wilson_upper(0, 100)
        self.assertTrue(0.0 < upper < 0.05)
        self.assertEqual(wilson_upper(0, 0), 1.0)

    def test_validate_payload(self):
        valid = {"n_test": 10, "candidate_indices": [1, 2, 5], "contains_test_labels": False, "contains_evaluation_metrics": False}
        validate_prediction_payload(valid)

        with self.assertRaises(ValueError):
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1], "contains_test_labels": False, "contains_evaluation_metrics": False, "y_test": [0]})

        with self.assertRaises(ValueError):
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1], "contains_evaluation_metrics": False})
            
        with self.assertRaises(ValueError):
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1, 1], "contains_test_labels": False, "contains_evaluation_metrics": False})

        with self.assertRaises(ValueError):
            validate_prediction_payload({"n_test": 10, "candidate_indices": [10], "contains_test_labels": False, "contains_evaluation_metrics": False})


class Exp206210AlternativeOOFTests(unittest.TestCase):

    def test_query_values_do_not_change_fit_scores(self):
        # Create synthetic time series data
        np.random.seed(42)
        train = np.random.randn(20, 100)
        test = np.random.randn(10, 100)

        for source in ["spectral", "shapelet", "interval"]:
            fit_a, query_a = fit_score_source(source, train, test, seed=7)
            changed_query = test.copy()
            changed_query[0] += np.sin(np.linspace(0, 20, test.shape[1]))
            fit_b, query_b = fit_score_source(source, train, changed_query, seed=7)

            np.testing.assert_allclose(fit_a, fit_b, err_msg=f"Fit scores changed for {source}")
            self.assertFalse(np.array_equal(query_a, query_b), f"Query scores did not change for {source}")

    def test_crossfit_completeness_and_determinism(self):
        np.random.seed(42)
        train = np.random.randn(15, 100)
        test = np.random.randn(5, 100)

        res_a = crossfit_alternative_sources(train, test, folds=3, seed=123)
        res_b = crossfit_alternative_sources(train, test, folds=3, seed=123)

        for key in ["spectral_train", "spectral_test", "shapelet_train", "shapelet_test", "interval_train", "interval_test"]:
            self.assertIn(key, res_a)
            self.assertEqual(len(res_a[key]), 15 if "train" in key else 5)
            self.assertTrue(np.all(np.isfinite(res_a[key])))
            np.testing.assert_array_equal(res_a[key], res_b[key])
