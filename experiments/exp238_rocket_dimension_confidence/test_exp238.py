from __future__ import annotations

import unittest
import numpy as np

from experiments.exp238_rocket_dimension_confidence.core import build_confidence_lanes
from experiments.exp238_rocket_dimension_confidence.run_experiment_238 import _reference


class ConfidenceLaneTests(unittest.TestCase):
    def test_confidence_and_review_partitions_preserve_frozen_lanes(self) -> None:
        result = build_confidence_lanes(
            hard={1, 2, 3, 4},
            standard={5, 6, 7},
            priority={8},
            rocket_256={1, 2, 5, 6, 9},
            rocket_512={1, 2, 5, 9},
            rocket_1024={1, 2, 5, 9},
            rocket_256_seed2={1, 5, 9},
            independent_auxiliary={1, 3, 5, 6, 10},
        )
        self.assertEqual(result.confidence_a, frozenset({1}))
        self.assertEqual(result.confidence_b, frozenset({3}))
        self.assertEqual(result.confidence_c, frozenset({2, 4}))
        self.assertEqual(result.review_priority_1, frozenset({5}))
        self.assertEqual(result.review_priority_2, frozenset({6}))
        self.assertEqual(result.review_priority_3, frozenset({7}))
        self.assertEqual(result.confidence_a | result.confidence_b | result.confidence_c, frozenset({1,2,3,4}))
        self.assertEqual(result.review_priority_1 | result.review_priority_2 | result.review_priority_3, frozenset({5,6,7}))
        self.assertEqual(result.priority_unchanged, frozenset({8}))

    def test_correlated_rocket_dimensions_form_one_stable_family(self) -> None:
        result = build_confidence_lanes(
            hard={1}, standard=set(), priority=set(),
            rocket_256={1}, rocket_512={1}, rocket_1024=set(), rocket_256_seed2={1},
            independent_auxiliary=set(),
        )
        self.assertEqual(result.rocket_stable, frozenset())
        self.assertEqual(result.confidence_c, frozenset({1}))

    def test_reference_sampling_does_not_depend_on_dataset_name(self) -> None:
        values = np.arange(1200 * 8, dtype=np.float32).reshape(1200, 8)
        np.testing.assert_array_equal(
            _reference(values, 8, "original_name"),
            _reference(values, 8, "renamed_dataset"),
        )


if __name__ == "__main__":
    unittest.main()
