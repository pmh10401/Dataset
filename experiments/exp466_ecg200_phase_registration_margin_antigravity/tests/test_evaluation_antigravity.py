from __future__ import annotations

import dataclasses
from types import MappingProxyType
import unittest
from unittest.mock import patch


def qualification_antigravity(*, failed_period_equality: bool):
    from experiments.exp466_ecg200_phase_registration_margin_antigravity.contracts_antigravity import (
        Exp466FoldResultAntigravity,
        Exp466QualificationAntigravity,
    )

    rows = []
    for seed in (20260721, 20261721):
        for fold_index in range(5):
            failed = failed_period_equality and seed == 20260721 and fold_index == 0
            rows.append(
                Exp466FoldResultAntigravity(
                    seed=seed,
                    fold_index=fold_index,
                    per_type_auc=(0.90, 0.90, 0.90, 0.90 if failed else 0.95),
                    mean_auc=0.90 if failed else 0.9125,
                    c0_gradual_auc=0.90,
                    c0_period_auc=0.90,
                    c9_gradual_auc=0.90,
                    c9_period_auc=0.90 if failed else 0.95,
                    period_confirmations=1,
                    all_scores_finite=True,
                    passed=not failed,
                    failure_reasons=(
                        ("period_auc_not_strictly_above_c0",)
                        if failed
                        else ()
                    ),
                ),
            )
    return Exp466QualificationAntigravity(
        fold_results=tuple(rows),
        normal_trial_counts=MappingProxyType({20260721: 69, 20261721: 69}),
        normal_confirmations=MappingProxyType({20260721: 0, 20261721: 0}),
        label_query_count=0,
        official_evaluate_count=0,
        official_inspect_count=0,
        artifact_map_sha256=None,
    )


class EvaluationAntigravityTest(unittest.TestCase):
    def test_pure_c9_decision_accepts_ten_of_ten_without_state_changes_antigravity(self) -> None:
        """Break caught: evaluation rescoring or mutating routing/governance state."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            QualificationDecisionAntigravity,
            evaluate_qualification_antigravity,
        )

        qualification = qualification_antigravity(failed_period_equality=False)
        with patch(
            "experiments.exp466_ecg200_phase_registration_margin_antigravity."
            "qualification_antigravity.qualify_antigravity",
            side_effect=AssertionError("evaluation must not rescore"),
        ):
            decision = evaluate_qualification_antigravity(qualification)
        self.assertIs(type(decision), QualificationDecisionAntigravity)
        self.assertEqual(decision.status, "qualification_passed_awaiting_exp467_design")
        self.assertEqual(decision.primary_candidate, "C9")
        self.assertTrue(decision.eligible_for_exp467_design)
        self.assertEqual((decision.primary_passed_count, decision.primary_required_count), (10, 10))
        self.assertEqual(decision.failure_reasons, ())
        self.assertFalse(decision.routing_applied)
        self.assertFalse(decision.champion_modified)
        self.assertFalse(decision.pending_modified)
        self.assertFalse(decision.promotion_applied)
        self.assertFalse(decision.test_labels_opened)
        self.assertEqual(decision.label_query_count, 0)
        self.assertEqual(decision, evaluate_qualification_antigravity(qualification))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decision.status = "changed"  # type: ignore[misc]

    def test_failed_fold_blocks_exp467_with_deterministic_reasons_antigravity(self) -> None:
        """Break caught: a period-equality fold or non-10/10 result is eligible."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )

        decision = evaluate_qualification_antigravity(
            qualification_antigravity(failed_period_equality=True),
        )
        self.assertEqual(
            decision.status,
            "qualification_failed_blocked_before_fn_validation",
        )
        self.assertFalse(decision.eligible_for_exp467_design)
        self.assertEqual(decision.primary_passed_count, 9)
        self.assertEqual(
            decision.failure_reasons,
            ("c9_not_10_of_10", "period_auc_not_strictly_above_c0"),
        )

    def test_decision_contract_rejects_forged_eligible_nine_of_ten_antigravity(self) -> None:
        """Break caught: an eligible empty-reason decision claims only nine passing folds."""
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
            evaluate_qualification_antigravity,
        )

        valid = evaluate_qualification_antigravity(
            qualification_antigravity(failed_period_equality=False),
        )
        with self.assertRaisesRegex(ValueError, "eligible decision requires ten"):
            dataclasses.replace(valid, primary_passed_count=9)


if __name__ == "__main__":
    unittest.main()
