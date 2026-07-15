from __future__ import annotations

import unittest

import numpy as np

from experiments.exp167_168.parity_core import (
    ALPHAS,
    assert_label_free_prediction,
    candidate_rows,
    evaluate_indices,
    fold_paired_p_values,
)


class FoldPairedPValueTests(unittest.TestCase):
    def test_uses_matching_fold_and_finite_sample_correction(self):
        heldout = np.array([1.0, 5.0, 3.0])
        test = np.array([[2.0, 2.0], [4.0, 6.0], [3.0, 2.0]])

        actual = fold_paired_p_values(heldout, test)

        np.testing.assert_allclose(actual, np.array([3.0 / 4.0, 2.0 / 4.0]))

    def test_fold_order_does_not_change_p_values(self):
        heldout = np.array([1.0, 5.0, 3.0])
        test = np.array([[2.0, 2.0], [4.0, 6.0], [3.0, 2.0]])
        order = np.array([2, 0, 1])

        np.testing.assert_allclose(
            fold_paired_p_values(heldout, test),
            fold_paired_p_values(heldout[order], test[order]),
        )

    def test_rejects_shape_mismatch_and_non_finite_values(self):
        with self.assertRaises(ValueError):
            fold_paired_p_values(np.array([1.0, 2.0]), np.ones((3, 4)))
        with self.assertRaises(ValueError):
            fold_paired_p_values(np.array([1.0, np.nan]), np.ones((2, 4)))


class CandidateGridTests(unittest.TestCase):
    def setUp(self):
        self.p_values = {
            "spectral": np.array([0.001, 0.04, 0.90]),
            "shapelet": np.array([0.002, 0.80, 0.90]),
            "interval": np.array([0.003, 0.03, 0.90]),
        }

    def test_emits_all_fixed_source_configs_and_alphas(self):
        rows = candidate_rows(
            dataset="demo",
            n_train=9,
            n_test=3,
            p_values=self.p_values,
            config_hash="abc",
        )

        self.assertEqual(len(rows), len(ALPHAS) * 5)
        self.assertEqual(
            {row["source_config"] for row in rows},
            {
                "spectral+shapelet",
                "spectral+interval",
                "shapelet+interval",
                "2of3",
                "3of3",
            },
        )
        self.assertEqual({row["alpha"] for row in rows}, set(ALPHAS))
        for row in rows:
            assert_label_free_prediction(row)

    def test_candidate_logic_is_exact(self):
        rows = candidate_rows(
            dataset="demo",
            n_train=19,
            n_test=3,
            p_values=self.p_values,
            config_hash="abc",
        )
        by_key = {(row["alpha"], row["source_config"]): row for row in rows}

        self.assertEqual(by_key[(0.05, "spectral+shapelet")]["candidate_indices"], [0])
        self.assertEqual(by_key[(0.05, "spectral+interval")]["candidate_indices"], [0, 1])
        self.assertEqual(by_key[(0.05, "shapelet+interval")]["candidate_indices"], [0])
        self.assertEqual(by_key[(0.05, "2of3")]["candidate_indices"], [0, 1])
        self.assertEqual(by_key[(0.05, "3of3")]["candidate_indices"], [0])

    def test_prediction_rows_reject_evaluation_fields_and_bad_indices(self):
        row = candidate_rows(
            dataset="demo",
            n_train=9,
            n_test=3,
            p_values=self.p_values,
            config_hash="abc",
        )[0]
        row["tp"] = 1
        with self.assertRaises(ValueError):
            assert_label_free_prediction(row)

        row.pop("tp")
        row["candidate_indices"] = [0, 0]
        row["candidate_count"] = 2
        with self.assertRaises(ValueError):
            assert_label_free_prediction(row)

        row["candidate_indices"] = [3]
        row["candidate_count"] = 1
        with self.assertRaises(ValueError):
            assert_label_free_prediction(row)

    def test_evaluation_is_separate(self):
        metrics = evaluate_indices([1, 3], np.array([0, 1, 0, 1, 0]))
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
