import unittest

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    ConfusionAntigravity,
    DatasetBaselineAntigravity,
    DatasetBundleAntigravity,
    MethodCandidateAntigravity,
    MethodAuditAntigravity,
)


class ContractsAntigravityTest(unittest.TestCase):
    def test_bool_is_not_an_integer_metric_antigravity(self):
        with self.assertRaisesRegex(ValueError, "fn must be an exact integer"):
            ConfusionAntigravity(tp=1, fp=0, fn=True, tn=3)

    def test_contracts_are_frozen_slots_and_reject_coercions_antigravity(self):
        candidate = MethodCandidateAntigravity("m", "r", (1,), (0.005,))
        self.assertEqual(candidate.name, "m")
        self.assertFalse(hasattr(candidate, "__dict__"))
        with self.assertRaisesRegex(ValueError, "seeds must be an exact tuple"):
            MethodCandidateAntigravity("m", "r", [1], (0.005,))
        with self.assertRaisesRegex(ValueError, "qualified must be an exact bool"):
            CandidateMetricAntigravity("F", "m", ConfusionAntigravity(1, 0, 0, 1), ConfusionAntigravity(1, 0, 0, 1), 0, 1.0, 1)

    def test_numpy_bundle_is_exact_and_immutable_antigravity(self):
        train = np.array([[1.0]], dtype=np.float64)
        test = np.array([[2.0]], dtype=np.float64)
        bundle = DatasetBundleAntigravity("d", train, test, 1)
        self.assertFalse(bundle.train_normal.flags.writeable)
        with self.assertRaisesRegex(ValueError, "target_length must be an exact integer"):
            DatasetBundleAntigravity("d", train, test, True)

    def test_baseline_and_audit_exact_types_antigravity(self):
        confusion = ConfusionAntigravity(1, 0, 0, 1)
        baseline = DatasetBaselineAntigravity(0, "Crop", frozenset({1}), confusion)
        audit = MethodAuditAntigravity("MiniRocket", "score_compatible", "m", "s", "1", "a" * 64, ("fit",))
        self.assertEqual(baseline.dataset_order, 0)
        self.assertEqual(audit.methods, ("fit",))
        with self.assertRaisesRegex(ValueError, "methods must be an exact tuple"):
            MethodAuditAntigravity("x", "unavailable", "m", "s", None, None, ["fit"])


if __name__ == "__main__":
    unittest.main()
