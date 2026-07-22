from __future__ import annotations

import unittest

from experiments.exp309_316_fn_recovery.evaluation import _variant_fields


class EvaluationTests(unittest.TestCase):
    def test_source_and_fusion_variants_are_fixed(self):
        self.assertEqual(len(_variant_fields(309)), 4)
        self.assertEqual(len(_variant_fields(312)), 4)
        self.assertEqual(len(_variant_fields(316)), 16)
        self.assertIn(("alpha=0.005", "alpha_0p005_indices"), _variant_fields(309))


if __name__ == "__main__": unittest.main()
