import unittest

import numpy as np

from experiments.exp411_near_repeat_qualification.distances import (
    cpu_self_excluded_rms,
    calibrate_thresholds,
    cpu_pairwise_rms,
    tag_queries,
    view_matrices,
)


class DistanceTests(unittest.TestCase):
    def test_chunked_self_excluded_rms_matches_direct_definition(self) -> None:
        rows = np.asarray([[0.0, 0.0], [3.0, 4.0], [4.0, 4.0]])
        np.testing.assert_allclose(
            cpu_self_excluded_rms(rows, chunk_rows=2),
            np.asarray([np.sqrt(12.5), np.sqrt(0.5), np.sqrt(0.5)]),
            atol=1e-12,
        )
    def test_views_have_expected_widths_and_finite_values(self) -> None:
        rows = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.1, 2.0, 3.0]])
        views = view_matrices(rows)
        self.assertEqual(views["raw_rms"].shape, (2, 4))
        self.assertEqual(views["diff_rms"].shape, (2, 3))
        self.assertEqual(views["spectrum_rms"].shape, (2, 2))
        self.assertTrue(
            all(np.isfinite(values).all() for values in views.values())
        )

    def test_pairwise_rms_returns_each_query_minimum(self) -> None:
        fit = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        query = np.asarray([[0.0, 0.0], [4.0, 4.0]])
        np.testing.assert_allclose(
            cpu_pairwise_rms(fit, query),
            np.asarray([0.0, np.sqrt(0.5)]),
        )

    def test_near_requires_raw_and_two_available_views(self) -> None:
        thresholds = {
            "raw_rms": 0.2,
            "diff_rms": 0.2,
            "spectrum_rms": 0.2,
        }
        distances = {
            "raw_rms": np.asarray([0.1, 0.1]),
            "diff_rms": np.asarray([0.1, 0.3]),
            "spectrum_rms": np.asarray([0.3, 0.1]),
        }
        self.assertEqual(
            tag_queries(distances, thresholds, exact=(False, False)),
            ("near", "near"),
        )
        self.assertEqual(
            tag_queries(distances, thresholds, exact=(True, False)),
            ("exact", "near"),
        )

    def test_lower_quantile_is_frozen(self) -> None:
        values = {"raw_rms": np.asarray([1.0, 2.0, 3.0, 100.0])}
        self.assertEqual(calibrate_thresholds(values, 0.10)["raw_rms"], 1.0)

    def test_two_unavailable_views_abstain(self) -> None:
        self.assertEqual(
            tag_queries(
                {"raw_rms": np.asarray([0.0])},
                {"raw_rms": 0.1},
                exact=(False,),
            ),
            ("abstain",),
        )


if __name__ == "__main__":
    unittest.main()
