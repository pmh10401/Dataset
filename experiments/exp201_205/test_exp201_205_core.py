from __future__ import annotations

import unittest

import numpy as np

from experiments.exp201_205.core import (
    ALPHA,
    classify_size_stratum,
    cross_confirmed_candidates,
    empirical_tail_candidates,
    three_source_agreement,
    validate_prediction_payload,
)


class Exp201205CoreTests(unittest.TestCase):
    def test_frozen_alpha(self):
        self.assertEqual(ALPHA, 0.01)

    def test_size_strata(self):
        self.assertIsNone(classify_size_stratum(99))
        self.assertEqual(classify_size_stratum(100), "100-199")
        self.assertEqual(classify_size_stratum(199), "100-199")
        self.assertEqual(classify_size_stratum(200), "200+")

    def test_empirical_tail_is_strictly_above_higher_quantile(self):
        train = np.arange(100, dtype=float)
        test = np.asarray([98.0, 99.0, 100.0])
        self.assertEqual(empirical_tail_candidates(train, test, ALPHA), {2})

    def test_three_source_agreement_is_exact_intersection(self):
        result = three_source_agreement({1, 2, 3}, {2, 3, 4}, {0, 2, 3})
        self.assertEqual(result, {2, 3})

    def test_cross_confirmation_requires_exp198_and_excludes_exp137(self):
        result = cross_confirmed_candidates(
            alternative_indices={1, 2, 3},
            exp198_indices={2, 3, 4},
            excluded_exp137_indices={3},
        )
        self.assertEqual(result, {2})

    def test_prediction_payload_rejects_metrics_and_invalid_indices(self):
        base = {
            "dataset": "x",
            "n_test": 3,
            "candidate_indices": [1],
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        validate_prediction_payload(base)
        with self.assertRaises(ValueError):
            validate_prediction_payload({**base, "hard_tp": 1})
        with self.assertRaises(ValueError):
            validate_prediction_payload({**base, "candidate_indices": [3]})


if __name__ == "__main__":
    unittest.main()
