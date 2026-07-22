from __future__ import annotations

import dataclasses
import importlib.metadata
from pathlib import Path
import unittest

from experiments.exp267_273_zero_support_fn_recovery.config import (
    ALPHAS,
    CALIBRATION_ABSTENTION_REASON,
    CALIBRATION_MIN_TRAIN,
    DB_PATH,
    HARD_RISK_LIMIT,
    MIN_RETROSPECTIVE_PRECISION,
    MIN_TRAIN_FOR_CALIBRATION,
    PRIMARY_ALPHA,
    REVIEW_RISK_LIMIT,
    RFF_MAX_TRAIN_RESIDUAL_VALUES,
    RFF_MIN_TRAIN,
    SECOND_SEED,
    SEED,
    TCN_MAX_TRAIN_ELEMENTS,
    TCN_MIN_TRAIN,
)
from experiments.exp267_273_zero_support_fn_recovery.contracts import (
    FrozenSourcePrediction,
    PromotionAudit,
    RecoveryLanes,
    SourceSpec,
    SourceScore,
)
from experiments.exp267_273_zero_support_fn_recovery.dependencies import (
    dependency_report,
)


class ContractTest(unittest.TestCase):
    def test_frozen_policy_constants(self) -> None:
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(PRIMARY_ALPHA, 0.005)
        self.assertEqual(SEED, 20260721)
        self.assertEqual(SECOND_SEED, 20261721)
        self.assertEqual(RFF_MIN_TRAIN, 20)
        self.assertEqual(RFF_MAX_TRAIN_RESIDUAL_VALUES, 10_000_000)
        self.assertEqual(TCN_MIN_TRAIN, 100)
        self.assertEqual(TCN_MAX_TRAIN_ELEMENTS, 5_000_000)
        self.assertEqual(CALIBRATION_MIN_TRAIN, 5)
        self.assertEqual(MIN_TRAIN_FOR_CALIBRATION, 5)
        self.assertEqual(CALIBRATION_ABSTENTION_REASON, "n_train_below_5")
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

    def test_db_path_is_frozen(self) -> None:
        self.assertEqual(DB_PATH, Path("/Users/minho/Documents/Dataset/univariate_ts.db"))

    def test_dependency_report_matches_installed_versions(self) -> None:
        report = dependency_report()
        self.assertEqual(report["kymatio"], importlib.metadata.version("kymatio"))
        self.assertEqual(report["iisignature"], importlib.metadata.version("iisignature"))
        self.assertEqual(report["torch"], importlib.metadata.version("torch"))
        self.assertIsInstance(report["mps_built"], bool)
        self.assertIsInstance(report["mps_available"], bool)

    def test_source_score_provenance_is_deeply_read_only(self) -> None:
        provenance = {"nested": {"values": [1, 2]}}
        score = SourceScore(provenance=provenance)
        provenance["nested"]["values"].append(3)  # type: ignore[index]
        self.assertEqual(score.provenance["nested"]["values"], (1, 2))  # type: ignore[index]
        with self.assertRaises(TypeError):
            score.provenance["nested"]["values"] = ()  # type: ignore[index]

    def test_prediction_candidate_indices_are_defensively_sorted_tuples(self) -> None:
        candidates = [3, 1]
        prediction = FrozenSourcePrediction(
            dataset="demo",
            source="scattering",
            variant="primary",
            alpha=PRIMARY_ALPHA,
            candidate_indices=candidates,  # type: ignore[arg-type]
            score_hash="score-hash",
            eligibility=True,
            abstention_reason=None,
            config_hash="config-hash",
        )
        candidates.append(9)
        self.assertEqual(prediction.candidate_indices, (1, 3))

    def test_recovery_lanes_require_all_fields_and_normalize_indices(self) -> None:
        with self.assertRaises(TypeError):
            RecoveryLanes()
        lanes = RecoveryLanes(
            hard_indices=[3, 1],
            shadow_indices=[],
            priority_indices=[5],
            standard_indices=[8, 7],
            recovery_shadow_indices=[10],
            recovery_priority_indices=[12, 11],
        )
        self.assertEqual(lanes.hard_indices, (1, 3))
        self.assertEqual(lanes.standard_indices, (7, 8))
        self.assertEqual(lanes.recovery_priority_indices, (11, 12))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            lanes.hard_indices += (4,)

    def test_recovery_lanes_reject_pairwise_overlap(self) -> None:
        with self.assertRaises(ValueError):
            RecoveryLanes(
                hard_indices=[1],
                shadow_indices=[1],
                priority_indices=[],
                standard_indices=[],
                recovery_shadow_indices=[],
                recovery_priority_indices=[],
            )

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
                "data_quality_review_indices",
            },
        )


if __name__ == "__main__":
    unittest.main()
