from __future__ import annotations

import unittest

import numpy as np

from experiments.exp215_217.core import (
    central_normal_medoid_dictionary,
    fixed_kernel_dictionary,
    fit_shape_source,
)


class ShapeletStabilityCoreTests(unittest.TestCase):
    def test_fixed_dictionary_is_data_independent_and_repeatable(self) -> None:
        left = fixed_kernel_dictionary()
        right = fixed_kernel_dictionary()
        self.assertEqual(len(left), 32)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(left, right)))
        self.assertEqual(sorted({len(value) for value in left}), [8, 16, 32, 64])

    def test_central_medoid_dictionary_is_deterministic(self) -> None:
        normal = np.vstack([np.sin(np.linspace(0, 6, 96) + offset) for offset in np.linspace(0, 1, 12)]).astype(np.float32)
        left = central_normal_medoid_dictionary(normal)
        right = central_normal_medoid_dictionary(normal)
        self.assertEqual(len(left), 32)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(left, right)))

    def test_fit_shape_source_uses_fit_rows_only_for_medoid_dictionary(self) -> None:
        fit = np.vstack([np.sin(np.linspace(0, 6, 96) + offset) for offset in np.linspace(0, 1, 12)]).astype(np.float32)
        query = np.vstack([np.cos(np.linspace(0, 6, 96) + offset) for offset in np.linspace(0, 1, 3)]).astype(np.float32)
        _, first = fit_shape_source("normal_consensus_medoid", fit, query, seed=7)
        changed_query = query.copy()
        changed_query[0] *= 1000
        _, second = fit_shape_source("normal_consensus_medoid", fit, changed_query, seed=7)
        self.assertTrue(np.allclose(first[1:], second[1:]))


if __name__ == "__main__":
    unittest.main()
