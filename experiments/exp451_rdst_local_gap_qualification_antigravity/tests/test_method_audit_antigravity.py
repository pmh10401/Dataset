"""Subprocess audit of the frozen official aeon RDST surface."""

from __future__ import annotations

import unittest

from experiments.exp451_rdst_local_gap_qualification_antigravity.method_audit_antigravity import (
    RdstAuditResultAntigravity,
    audit_rdst_antigravity,
)


class MethodAuditAntigravityTest(unittest.TestCase):
    def test_fixed_interpreter_audit_binds_source_signature_and_label_free_fixture_antigravity(self) -> None:
        result = audit_rdst_antigravity()
        self.assertIs(type(result), RdstAuditResultAntigravity)
        self.assertEqual(result.aeon_version, "1.4.0")
        self.assertEqual(result.class_path, "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform")
        self.assertEqual(result.source_sha256, "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2")
        self.assertEqual(result.constructor_signature, "(max_shapelets: int = 10000, shapelet_lengths: List[int] | numpy.ndarray | None = None, proba_normalization: float = 0.8, threshold_percentiles: List[float] | numpy.ndarray | None = None, alpha_similarity: float = 0.5, use_prime_dilations: bool = False, random_state: int | None = None, n_jobs: int = 1)")
        self.assertEqual(result.seed, 20260721)
        self.assertEqual(result.max_shapelets, 1024)
        self.assertTrue(result.y_is_none)
        self.assertTrue(result.deterministic)
        self.assertGreater(result.retained_shapelets, 0)
        self.assertEqual(result.feature_dimension, 3 * result.retained_shapelets)


if __name__ == "__main__":
    unittest.main()
