from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import (
    empirical_candidates,
)
from experiments.exp239_248_diverse_score_sources.foundation import (
    deterministic_folds,
    load_bundle,
)


class FoundationContractTest(unittest.TestCase):
    def test_fold_assignment_is_deterministic(self) -> None:
        first = tuple(map(tuple, deterministic_folds(23, seed=20260720)))
        second = tuple(map(tuple, deterministic_folds(23, seed=20260720)))
        self.assertEqual(first, second)
        self.assertEqual(sorted(value for fold in first for value in fold), list(range(23)))

    def test_empirical_candidates_reports_resolution(self) -> None:
        result = empirical_candidates(np.arange(9.0), np.asarray([20.0]), 0.005)
        self.assertEqual(result.attainable_p, 0.1)
        self.assertTrue(result.resolution_blocked)
        self.assertEqual(result.indices, ())

    def test_loader_source_does_not_select_test_labels(self) -> None:
        source = inspect.getsource(load_bundle).lower()
        self.assertNotIn("select label", source)
        self.assertNotIn("select i.label", source)


if __name__ == "__main__":
    unittest.main()
