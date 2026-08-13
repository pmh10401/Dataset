from __future__ import annotations

import unittest

import numpy as np

from experiments.exp451_rdst_local_gap_qualification_antigravity.rdst_antigravity import (
    RDSTRepresentationAntigravity,
    make_rdst_antigravity,
)


class RDSTAntigravityTest(unittest.TestCase):
    def test_official_wrapper_has_finite_exact_width_and_retained_property_antigravity(self) -> None:
        rows = np.random.default_rng(451).normal(size=(5, 1, 32)).astype(np.float64)
        representation = make_rdst_antigravity(20260721)
        self.assertIs(type(representation), RDSTRepresentationAntigravity)
        with self.assertRaisesRegex(RuntimeError, "fit"):
            _ = representation.retained_shapelets
        features = representation.fit_transform(rows, y=None)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape[1], 3 * representation.retained_shapelets)

    def test_wrapper_rejects_labels_and_invalid_or_incompatible_rows_antigravity(self) -> None:
        valid = np.random.default_rng(452).normal(size=(5, 1, 32)).astype(np.float64)
        representation = make_rdst_antigravity(20260721)
        with self.assertRaises(ValueError):
            representation.fit(valid, y=np.zeros(5))
        with self.assertRaises(ValueError):
            representation.transform(valid)
        for bad in (True, np.ones((5, 32)), np.full((5, 1, 32), np.nan), np.ones((0, 1, 32))):
            with self.subTest(bad=repr(bad)), self.assertRaises(ValueError):
                representation.fit(bad, y=None)
        representation.fit(valid, y=None)
        with self.assertRaises(ValueError):
            representation.transform(np.ones((2, 1, 31), dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
