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
        self.assertEqual(bundle.train_normal.dtype, train.dtype)
        self.assertEqual(bundle.train_normal.shape, train.shape)
        train[0, 0] = 99.0
        self.assertEqual(bundle.train_normal[0, 0], 1.0)
        with self.assertRaises(ValueError):
            bundle.train_normal.setflags(write=True)
        with self.assertRaises(ValueError):
            bundle.test_values.setflags(write=True)
        with self.assertRaisesRegex(ValueError, "target_length must be an exact integer"):
            DatasetBundleAntigravity("d", train, test, True)

    def test_all_public_string_fields_reject_subclasses_antigravity(self):
        class StringSubclassAntigravity(str):
            pass

        text = StringSubclassAntigravity("text")
        confusion = ConfusionAntigravity(1, 0, 0, 1)
        cases = (
            ("candidate name", lambda: MethodCandidateAntigravity(text, "r", (1,), (0.005,))),
            ("candidate representation", lambda: MethodCandidateAntigravity("m", text, (1,), (0.005,))),
            ("audit name", lambda: MethodAuditAntigravity(text, "unavailable", "m", "s", None, None, ())),
            ("audit status", lambda: MethodAuditAntigravity("n", StringSubclassAntigravity("unavailable"), "m", "s", None, None, ())),
            ("audit module", lambda: MethodAuditAntigravity("n", "unavailable", text, "s", None, None, ())),
            ("audit symbol", lambda: MethodAuditAntigravity("n", "unavailable", "m", text, None, None, ())),
            ("audit package", lambda: MethodAuditAntigravity("n", "unavailable", "m", "s", text, None, ())),
            ("audit source", lambda: MethodAuditAntigravity("n", "score_compatible", "m", "s", None, StringSubclassAntigravity("a" * 64), ("fit",))),
            ("audit method", lambda: MethodAuditAntigravity("n", "score_compatible", "m", "s", None, "a" * 64, (text,))),
            ("bundle dataset", lambda: DatasetBundleAntigravity(text, np.array([[1.0]]), np.array([[2.0]]), 1)),
            ("baseline family", lambda: DatasetBaselineAntigravity(0, text, frozenset({1}), confusion)),
            ("metric family", lambda: CandidateMetricAntigravity(text, "m", confusion, confusion, 0, 1.0, True)),
            ("metric method", lambda: CandidateMetricAntigravity("F", text, confusion, confusion, 0, 1.0, True)),
        )
        for label, construct in cases:
            with self.subTest(label=label), self.assertRaises(ValueError):
                construct()

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
