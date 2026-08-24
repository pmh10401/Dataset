from __future__ import annotations

import dataclasses
from types import MappingProxyType
import unittest


def _qualification_antigravity(**changes):
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.contracts_antigravity import (
        Exp470FoldResultAntigravity, Exp470QualificationAntigravity, PairRepairCountsAntigravity,
    )
    types = ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
    rows = tuple(
        Exp470FoldResultAntigravity(
            seed=seed, fold_index=fold, c0_per_type_auc=dict.fromkeys(types, 0.10), c11_per_type_auc=dict.fromkeys(types, 0.10),
            pair_repairs={name: PairRepairCountsAntigravity(1 if name == "period_change" else 0, 0) for name in types},
            normal_confirmations=0, fallback_exact=True, finite=True, passed=True, failure_reasons=(),
        ) for seed in (20260721, 20261721) for fold in range(5)
    )
    values = dict(fold_results=rows, period_fixed={20260721: 5, 20261721: 5}, period_broken={20260721: 0, 20261721: 0}, protected_broken={"level_shift": 0, "gradual_drift": 0, "amplitude_scaling": 0}, normal_confirmations={20260721: 0, 20261721: 0}, fallback_exact=True, finite=True, label_query_count=0, artifact_map_sha256=None)
    values.update(changes)
    return Exp470QualificationAntigravity(**values)


class Exp470EvaluationTestAntigravity(unittest.TestCase):
    def test_pair_gates_pass_even_when_auc_is_below_085_antigravity(self) -> None:
        """Break caught: observed AUC 0.85 becomes an unauthorized decision gate."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.evaluation_antigravity import evaluate_repair_gates_antigravity

        decision = evaluate_repair_gates_antigravity(_qualification_antigravity())
        self.assertIs(decision.passed, True)
        self.assertIs(decision.eligible_for_private_reproducibility, True)
        self.assertEqual(decision.failure_reasons, ())

    def test_each_pair_gate_failure_has_one_stable_reason_antigravity(self) -> None:
        """Break caught: an aggregate repair, fallback, or finite failure is silently ignored."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.evaluation_antigravity import evaluate_repair_gates_antigravity

        cases = (
            (dict(period_fixed={20260721: 0, 20261721: 5}), "seed_20260721_period_fixed_below_1"),
            (dict(period_broken={20260721: 1, 20261721: 0}), "seed_20260721_period_broken_not_zero"),
            (dict(protected_broken={"level_shift": 1, "gradual_drift": 0, "amplitude_scaling": 0}), "level_shift_broken_not_zero"),
            (dict(normal_confirmations={20260721: 1, 20261721: 0}), "seed_20260721_normal_confirmations_not_zero"),
            (dict(fallback_exact=False), "fallback_not_exact"),
            (dict(finite=False), "nonfinite_scores"),
        )
        for changes, expected in cases:
            with self.subTest(expected=expected):
                decision = evaluate_repair_gates_antigravity(_qualification_antigravity(**changes))
                self.assertFalse(decision.passed)
                self.assertEqual(decision.failure_reasons, (expected,))

    def test_decision_is_immutable_and_cannot_forge_eligibility_antigravity(self) -> None:
        """Break caught: the decision's pass flag disagrees with its failure reasons."""
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.evaluation_antigravity import evaluate_repair_gates_antigravity

        decision = evaluate_repair_gates_antigravity(_qualification_antigravity())
        with self.assertRaises(ValueError):
            dataclasses.replace(decision, passed=False)


if __name__ == "__main__":
    unittest.main()
