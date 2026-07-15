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
        # 2.5: ref >= 2.5 is [3.0] (1 value). count = 1. p = (1 + 1) / (3 + 1) = 2/4 = 0.5
        # 0.5: ref >= 0.5 is [1.0, 2.0, 3.0] (3 values). count = 3. p = (3 + 1) / (3 + 1) = 1.0
        # 4.0: ref >= 4.0 is [] (0 values). count = 0. p = (0 + 1) / (3 + 1) = 0.25
        np.testing.assert_allclose(pvals, [0.5, 1.0, 0.25])

    def test_robust_z_pair(self):
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        query = np.array([0.0, 6.0])
        tr_z, q_z = robust_z_pair(train, query)
        self.assertEqual(len(tr_z), 5)
        self.assertEqual(len(q_z), 2)
        # Median is 3. MAD = median(|1-3|, |2-3|, |3-3|, |4-3|, |5-3|) = median(2, 1, 0, 1, 2) = 1.
        # Scale = 1.4826 * 1 = 1.4826
        # Expected tr_z = (train - 3.0) / 1.4826
        np.testing.assert_allclose(tr_z, (train - 3.0) / 1.4826)
        np.testing.assert_allclose(q_z, (query - 3.0) / 1.4826)

    def test_top2_mean(self):
        mat = np.array([
            [1.0, 4.0, 2.0],
            [10.0, 5.0, 8.0]
        ])
        res = top2_mean(mat)
        # row 0 top 2: 4.0 and 2.0 -> mean = 3.0
        # row 1 top 2: 10.0 and 8.0 -> mean = 9.0
        np.testing.assert_allclose(res, [3.0, 9.0])

    def test_inverse_spearman_weights(self):
        # Generate some dummy data
        np.random.seed(42)
        x = np.random.randn(100, 3)
        weights = inverse_spearman_weights(x)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(weights.mean(), 1.0)

    def test_fisher_stat(self):
        pvals = np.array([0.05, 0.01])
        stat = fisher_stat(pvals)
        # stat = -2 * (log(0.05) + log(0.01)) = -2 * (-2.995732 + -4.60517) = 15.2018
        self.assertAlmostEqual(stat, -2.0 * (np.log(0.05) + np.log(0.01)))

    def test_wilson_upper(self):
        # 0 successes out of 100 trials, 95% confidence one-sided
        upper = wilson_upper(0, 100)
        # Check that it is greater than 0 and less than 0.05 (specifically around 0.03)
        self.assertTrue(0.0 < upper < 0.05)
        # 0 trials returns 1.0
        self.assertEqual(wilson_upper(0, 0), 1.0)

    def test_validate_payload(self):
        valid = {"n_test": 10, "candidate_indices": [1, 2, 5], "contains_test_labels": False, "contains_evaluation_metrics": False}
        validate_prediction_payload(valid)  # Should not raise

        with self.assertRaises(ValueError):
            # Forbidden field
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1], "contains_test_labels": False, "contains_evaluation_metrics": False, "y_test": [0]})

        with self.assertRaises(ValueError):
            # Missing flag
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1], "contains_evaluation_metrics": False})
            
        with self.assertRaises(ValueError):
            # Duplicate indices
            validate_prediction_payload({"n_test": 10, "candidate_indices": [1, 1], "contains_test_labels": False, "contains_evaluation_metrics": False})

        with self.assertRaises(ValueError):
            # Out of bounds index
            validate_prediction_payload({"n_test": 10, "candidate_indices": [10], "contains_test_labels": False, "contains_evaluation_metrics": False})
