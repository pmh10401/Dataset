from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.calibration import (
    conformal_alerts,
    matched_top_k,
    upper_tail_p_values,
)


class CalibrationTests(unittest.TestCase):
    def test_upper_tail_p_values_use_finite_sample_correction(self) -> None:
        reference = np.asarray([1.0, 2.0, 3.0])
        query = np.asarray([0.0, 2.0, 4.0])

        actual = upper_tail_p_values(reference, query)

        np.testing.assert_array_equal(actual, np.asarray([1.0, 0.75, 0.25]))

    def test_conformal_alerts_use_less_than_or_equal_alpha(self) -> None:
        reference = np.arange(1.0, 100.0)
        query = np.asarray([99.0, 100.0, 101.0])

        alerts = conformal_alerts(
            reference, query, test_indices=(7, 8, 9), alpha=0.02
        )

        self.assertEqual(alerts, (7, 8, 9))

    def test_matched_top_k_breaks_score_ties_by_test_index(self) -> None:
        scores = np.asarray([0.5, 0.9, 0.9, 0.1])

        selected = matched_top_k(scores, indices=(5, 8, 3, 1), k=2)

        self.assertEqual(selected, (3, 8))

    def test_calibration_rejects_nonfinite_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            upper_tail_p_values(np.asarray([1.0]), np.asarray([np.nan]))


if __name__ == "__main__":
    unittest.main()
