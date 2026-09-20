"""Task 4 contracts for the scalar-only qualification boundary decision."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
    Exp468FoldResultAntigravity,
    Exp468QualificationAntigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.evaluation_antigravity import (
    QualificationDecisionAntigravity,
    evaluate_qualification_antigravity,
)


def qualification_fixture_antigravity(
    *,
    failed_fold: bool = False,
    seed_normal_confirmation: bool = False,
) -> Exp468QualificationAntigravity:
    rows: list[Exp468FoldResultAntigravity] = []
    for seed in (20260721, 20261721):
        for fold_index in range(5):
            failed = failed_fold and seed == 20260721 and fold_index == 0
            rows.append(
                Exp468FoldResultAntigravity(
                    seed=seed,
                    fold_index=fold_index,
                    c0_per_type_auc=(0.70, 0.70, 0.70, 0.90),
                    c10_per_type_auc=(0.70, 0.70, 0.70, 0.90 if failed else 0.95),
                    period_confirmations=1,
                    normal_confirmations=0,
                    all_scores_finite=True,
                    passed=not failed,
                    failure_reasons=(
                        ("period_auc_not_strictly_above_c0",)
                        if failed
                        else ()
                    ),
                ),
            )
    normal_confirmations = {
        20260721: 1 if seed_normal_confirmation else 0,
        20261721: 0,
    }
    return Exp468QualificationAntigravity(
        fold_results=tuple(rows),
        normal_trial_counts=MappingProxyType({20260721: 69, 20261721: 69}),
        normal_confirmations=MappingProxyType(normal_confirmations),
        label_query_count=0,
        artifact_map_sha256=None,
    )


class EvaluationAntigravityTest(unittest.TestCase):
    def test_ten_of_ten_scalar_decision_awaits_private_reproducibility_antigravity(
        self,
    ) -> None:
        """Break caught: Task 4 is mislabeled final PASS or mutates external state."""

        qualification = qualification_fixture_antigravity()
        with patch(
            "experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity."
            "qualification_antigravity.qualify_antigravity",
            side_effect=AssertionError("evaluation must not rescore"),
        ):
            decision = evaluate_qualification_antigravity(qualification)
        self.assertIs(type(decision), QualificationDecisionAntigravity)
        self.assertEqual(
            decision.status,
            "qualification_gate_passed_awaiting_private_reproducibility",
        )
        self.assertEqual(decision.primary_candidate, "C10")
        self.assertTrue(decision.eligible_for_private_reproducibility)
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

    def test_failed_fold_blocks_before_private_run_with_deterministic_reasons_antigravity(
        self,
    ) -> None:
        """Break caught: nine of ten folds can reach Task 5 private reproducibility."""

        decision = evaluate_qualification_antigravity(
            qualification_fixture_antigravity(failed_fold=True),
        )
        self.assertEqual(
            decision.status,
            "qualification_failed_blocked_before_private_run",
        )
        self.assertFalse(decision.eligible_for_private_reproducibility)
        self.assertEqual(decision.primary_passed_count, 9)
        self.assertEqual(
            decision.failure_reasons,
            ("c10_not_10_of_10", "period_auc_not_strictly_above_c0"),
        )

    def test_seed_normal_confirmation_blocks_even_with_ten_passing_rows_antigravity(
        self,
    ) -> None:
        """Break caught: aggregate seed normal confirmations are ignored."""

        decision = evaluate_qualification_antigravity(
            qualification_fixture_antigravity(seed_normal_confirmation=True),
        )
        self.assertFalse(decision.eligible_for_private_reproducibility)
        self.assertEqual(decision.primary_passed_count, 10)
        self.assertEqual(
            decision.failure_reasons,
            ("seed_20260721_normal_confirmations_not_zero",),
        )
        self.assertEqual(
            decision.status,
            "qualification_failed_blocked_before_private_run",
        )

    def test_decision_contract_rejects_forged_private_eligibility_antigravity(self) -> None:
        """Break caught: a nine-of-ten decision is forged as private-run eligible."""

        valid = evaluate_qualification_antigravity(qualification_fixture_antigravity())
        with self.assertRaisesRegex(ValueError, "eligible decision requires ten"):
            dataclasses.replace(valid, primary_passed_count=9)
        with self.assertRaisesRegex(ValueError, "exact Exp468Qualification"):
            evaluate_qualification_antigravity(object())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
