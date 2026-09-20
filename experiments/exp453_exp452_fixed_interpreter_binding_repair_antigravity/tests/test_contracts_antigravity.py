from __future__ import annotations

import unittest

import numpy as np

from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.contracts_antigravity import (
    CandidateMetricAntigravity,
    ConfusionAntigravity,
    DatasetBaselineAntigravity,
    DatasetBundleAntigravity,
    MethodAuditAntigravity,
    MethodCandidateAntigravity,
)


class ContractsAntigravityTest(unittest.TestCase):
    def test_bool_is_not_an_integer_metric_antigravity(self) -> None:
        """Break caught: Python bool silently entering a confusion count."""
        with self.assertRaisesRegex(ValueError, "fn must be an exact integer"):
            ConfusionAntigravity(tp=1, fp=0, fn=True, tn=3)

    def test_contracts_are_frozen_slots_and_reject_coercions_antigravity(self) -> None:
        """Break caught: mutable records or list/integer coercion changing authority."""
        candidate = MethodCandidateAntigravity("m", "r", (1,), (0.005,))
        self.assertEqual(candidate.name, "m")
        self.assertFalse(hasattr(candidate, "__dict__"))
        with self.assertRaisesRegex(ValueError, "seeds must be an exact tuple"):
            MethodCandidateAntigravity("m", "r", [1], (0.005,))  # type: ignore[arg-type]
        confusion = ConfusionAntigravity(1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "qualified must be an exact bool"):
            CandidateMetricAntigravity("F", "m", confusion, confusion, 0, 1.0, 1)  # type: ignore[arg-type]

    def test_numpy_bundle_is_exact_detached_and_immutable_antigravity(self) -> None:
        """Break caught: a caller mutating an array after validation."""
        train = np.array([[1.0]], dtype=np.float64)
        test = np.array([[2.0]], dtype=np.float64)
        bundle = DatasetBundleAntigravity("d", train, test, 1)
        self.assertFalse(bundle.train_normal.flags.writeable)
        self.assertEqual((bundle.train_normal.dtype, bundle.train_normal.shape), (train.dtype, train.shape))
        train[0, 0] = 99.0
        self.assertEqual(bundle.train_normal[0, 0], 1.0)
        with self.assertRaises(ValueError):
            bundle.train_normal.setflags(write=True)
        with self.assertRaisesRegex(ValueError, "target_length must be an exact integer"):
            DatasetBundleAntigravity("d", train, test, True)

    def test_all_public_string_fields_reject_subclasses_antigravity(self) -> None:
        """Break caught: string subclasses bypassing exact frozen contracts."""
        class StringSubclassAntigravity(str):
            pass

        text = StringSubclassAntigravity("text")
        confusion = ConfusionAntigravity(1, 0, 0, 1)
        cases = (
            lambda: MethodCandidateAntigravity(text, "r", (1,), (0.005,)),
            lambda: MethodCandidateAntigravity("m", text, (1,), (0.005,)),
            lambda: MethodAuditAntigravity(text, "unavailable", "m", "s", None, None, ()),
            lambda: MethodAuditAntigravity("n", StringSubclassAntigravity("unavailable"), "m", "s", None, None, ()),
            lambda: MethodAuditAntigravity("n", "unavailable", text, "s", None, None, ()),
            lambda: MethodAuditAntigravity("n", "unavailable", "m", text, None, None, ()),
            lambda: MethodAuditAntigravity("n", "score_compatible", "m", "s", None, StringSubclassAntigravity("a" * 64), ("fit",)),
            lambda: DatasetBundleAntigravity(text, np.array([[1.0]]), np.array([[2.0]]), 1),
            lambda: DatasetBaselineAntigravity(0, text, frozenset({1}), confusion),
            lambda: CandidateMetricAntigravity(text, "m", confusion, confusion, 0, 1.0, True),
            lambda: CandidateMetricAntigravity("F", text, confusion, confusion, 0, 1.0, True),
        )
        for construct in cases:
            with self.subTest(construct=construct), self.assertRaises(ValueError):
                construct()


if __name__ == "__main__":
    unittest.main()
