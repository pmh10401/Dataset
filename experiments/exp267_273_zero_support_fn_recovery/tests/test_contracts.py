from __future__ import annotations

import dataclasses
import unittest

from experiments.exp267_273_zero_support_fn_recovery.config import (
    ALPHAS,
    HARD_RISK_LIMIT,
    MIN_RETROSPECTIVE_PRECISION,
    PRIMARY_ALPHA,
    REVIEW_RISK_LIMIT,
    RFF_MIN_TRAIN,
    SECOND_SEED,
    SEED,
    TCN_MIN_TRAIN,
)
from experiments.exp267_273_zero_support_fn_recovery.contracts import (
    FrozenSourcePrediction,
    PromotionAudit,
    RecoveryLanes,
    SourceSpec,
    SourceScore,
)


class ContractTest(unittest.TestCase):
    def test_frozen_policy_constants(self) -> None:
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(PRIMARY_ALPHA, 0.005)
        self.assertEqual(SEED, 20260721)
        self.assertEqual(SECOND_SEED, 20261721)
        self.assertEqual(RFF_MIN_TRAIN, 20)
        self.assertEqual(TCN_MIN_TRAIN, 100)
        self.assertEqual(REVIEW_RISK_LIMIT, 0.01)
        self.assertEqual(HARD_RISK_LIMIT, 0.005)
        self.assertEqual(MIN_RETROSPECTIVE_PRECISION, 0.80)

    def test_prediction_contract_has_no_evaluation_fields(self) -> None:
        fields = {field.name for field in dataclasses.fields(FrozenSourcePrediction)}
        self.assertFalse(fields & {"y_test", "tp", "fp", "fn", "f1", "precision"})
        self.assertEqual(
            fields,
            {
                "dataset",
                "source",
                "variant",
                "alpha",
                "candidate_indices",
                "score_hash",
                "eligibility",
                "abstention_reason",
                "config_hash",
                "contains_test_labels",
            },
        )

    def test_contract_records_are_immutable(self) -> None:
        self.assertTrue(dataclasses.fields(SourceSpec))
        self.assertTrue(dataclasses.fields(SourceScore))
        self.assertTrue(dataclasses.fields(RecoveryLanes))
        self.assertTrue(dataclasses.fields(PromotionAudit))
        for record in (
            SourceSpec,
            SourceScore,
            FrozenSourcePrediction,
            RecoveryLanes,
            PromotionAudit,
        ):
            self.assertIsNotNone(record.__dataclass_params__)
            self.assertTrue(record.__dataclass_params__.frozen)

    def test_frozen_prediction_defaults_to_label_free(self) -> None:
        prediction = FrozenSourcePrediction(
            dataset="demo",
            source="scattering",
            variant="primary",
            alpha=PRIMARY_ALPHA,
            candidate_indices=(1, 4),
            score_hash="score-hash",
            eligibility=True,
            abstention_reason=None,
            config_hash="config-hash",
        )
        self.assertFalse(prediction.contains_test_labels)

    def test_recovery_lanes_have_immutable_existing_and_recovery_indices(self) -> None:
        fields = {field.name for field in dataclasses.fields(RecoveryLanes)}
        self.assertEqual(
            fields,
            {
                "hard_indices",
                "shadow_indices",
                "priority_indices",
                "standard_indices",
                "recovery_shadow_indices",
                "recovery_priority_indices",
            },
        )


if __name__ == "__main__":
    unittest.main()
