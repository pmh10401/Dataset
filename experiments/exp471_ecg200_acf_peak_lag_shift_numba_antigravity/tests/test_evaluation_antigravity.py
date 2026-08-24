from __future__ import annotations

import dataclasses
import unittest


class Exp471EvaluationTestAntigravity(unittest.TestCase):
    def valid_decision_antigravity(self, **changes):
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.evaluation_antigravity import (
            evaluate_repair_gates_antigravity,
        )

        values = {
            "period_fixed": {20260721: 1, 20261721: 1},
            "period_broken": {20260721: 0, 20261721: 0},
            "protected_broken": {"level_shift": 0, "gradual_drift": 0, "amplitude_scaling": 0},
            "normal_confirmations": {20260721: 0, 20261721: 0},
            "fallback_exact": True,
            "finite": True,
        }
        values.update(changes)
        return evaluate_repair_gates_antigravity(**values)

    def test_decision_requires_seedwise_period_repairs_antigravity(self) -> None:
        decision = self.valid_decision_antigravity()
        self.assertTrue(decision.passed)
        self.assertTrue(decision.eligible_for_private_reproducibility)
        self.assertEqual(decision.failure_reasons, ())

    def test_each_gate_has_stable_reason_and_no_auc_parameter_antigravity(self) -> None:
        cases = (
            ({"period_fixed": {20260721: 0, 20261721: 1}}, "period_fixed_zero:20260721"),
            ({"period_broken": {20260721: 1, 20261721: 0}}, "period_broken:20260721"),
            ({"protected_broken": {"level_shift": 1, "gradual_drift": 0, "amplitude_scaling": 0}}, "protected_broken:level_shift"),
            ({"normal_confirmations": {20260721: 0, 20261721: 1}}, "normal_confirmation:20261721"),
            ({"fallback_exact": False}, "fallback_not_exact"),
            ({"finite": False}, "nonfinite"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                decision = self.valid_decision_antigravity(**changes)
                self.assertFalse(decision.passed)
                self.assertEqual(decision.failure_reasons, (expected,))

    def test_decision_is_immutable_and_cannot_forge_eligibility_antigravity(self) -> None:
        decision = self.valid_decision_antigravity()
        with self.assertRaises(ValueError):
            dataclasses.replace(decision, passed=False)


if __name__ == "__main__":
    unittest.main()
