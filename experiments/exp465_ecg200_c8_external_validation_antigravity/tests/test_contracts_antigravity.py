from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType
import unittest


class ContractsAntigravityTest(unittest.TestCase):
    def _fold_row_antigravity(self, *, seed=20260721, fold_index=0, **changes):
        from experiments.exp465_ecg200_c8_external_validation_antigravity.contracts_antigravity import (
            Exp465FoldResultContractAntigravity,
        )

        values = {
            "seed": seed,
            "fold_index": fold_index,
            "per_type_auc": (0.9, 0.91, 0.92, 0.93),
            "mean_auc": 0.915,
            "c0_gradual_auc": 0.88,
            "c0_period_auc": 0.89,
            "c8_gradual_auc": 0.91,
            "c8_period_auc": 0.93,
            "all_scores_finite": True,
            "passed": True,
            "failure_reasons": (),
        }
        values.update(changes)
        return Exp465FoldResultContractAntigravity(**values)

    def test_contract_inventory_is_immutable_and_score_shaped(self) -> None:
        """Break caught: future score/result containers lose exact shape or immutability."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.contracts_antigravity import (
            Exp465MethodScoreContractAntigravity,
            Exp465QualificationContractAntigravity,
        )

        self.assertEqual(
            tuple(field.name for field in fields(Exp465MethodScoreContractAntigravity)),
            ("method", "scores", "confirmations", "evidence_sha256"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp465QualificationContractAntigravity)),
            (
                "rows",
                "seed_normal_confirmations",
                "label_query_count",
                "official_evaluate_count",
                "artifact_map_sha256",
            ),
        )
        score = Exp465MethodScoreContractAntigravity(
            method="C8",
            scores=(0.1, 0.2, 0.3),
            confirmations=(0, 0, 0),
            evidence_sha256="a" * 64,
        )
        self.assertIsInstance(score.scores, tuple)
        with self.assertRaises(FrozenInstanceError):
            score.method = "C6"  # type: ignore[misc]
        row = self._fold_row_antigravity()
        with self.assertRaises(FrozenInstanceError):
            row.passed = False  # type: ignore[misc]

    def test_method_score_rejects_type_aliases_and_nonfinite_scores(self) -> None:
        """Break caught: bool aliases or non-finite score values enter a method contract."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.contracts_antigravity import (
            Exp465MethodScoreContractAntigravity,
        )

        mutations = (
            {"method": 8},
            {"method": "C7"},
            {"scores": [True, False]},
            {"scores": (True, 0.2)},
            {"scores": (0.1, float("nan"))},
            {"confirmations": (False, True)},
            {"confirmations": [0, 0]},
            {"evidence_sha256": "A" * 64},
        )
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                values = {
                    "method": "C8",
                    "scores": (0.1, 0.2),
                    "confirmations": (0, 0),
                    "evidence_sha256": "b" * 64,
                }
                values.update(changes)
                Exp465MethodScoreContractAntigravity(**values)

    def test_fold_result_rejects_aliases_mean_drift_and_reason_mismatch(self) -> None:
        """Break caught: forged row aliases, mean drift, or passed/reasons mismatch."""
        mutations = (
            {"seed": True},
            {"seed": 20260722},
            {"fold_index": True},
            {"fold_index": 5},
            {"per_type_auc": [0.9, 0.91, 0.92, 0.93]},
            {"per_type_auc": (True, 0.91, 0.92, 0.93)},
            {"per_type_auc": (0.9, 0.91, float("inf"), 0.93)},
            {"per_type_auc": (0.9, 0.91, 0.92)},
            {"all_scores_finite": 1},
            {"passed": 1},
            {"failure_reasons": ["primary_not_10_of_10"]},
            {"failure_reasons": ("primary_not_10_of_10",)},
            {"passed": False, "failure_reasons": ()},
            {"passed": True, "failure_reasons": ("primary_not_10_of_10",)},
        )
        for changes in mutations:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self._fold_row_antigravity(**changes)
        with self.assertRaises(ValueError):
            self._fold_row_antigravity(per_type_auc=(0.9, 0.91, 0.92, 0.94), mean_auc=0.99)

    def test_qualification_freezes_rows_seed_map_and_initial_governance(self) -> None:
        """Break caught: qualification loses exact rows, seed order, or governance truth."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )
        from experiments.exp465_ecg200_c8_external_validation_antigravity.contracts_antigravity import (
            Exp465QualificationContractAntigravity,
            initial_governance_state_antigravity,
        )

        rows = tuple(
            self._fold_row_antigravity(seed=seed, fold_index=fold)
            for seed in FROZEN_CONFIG_ANTIGRAVITY.seeds
            for fold in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
        )
        contract = Exp465QualificationContractAntigravity(
            rows=rows,
            seed_normal_confirmations={20260721: 0, 20261721: 0},
            label_query_count=0,
            official_evaluate_count=0,
            artifact_map_sha256=None,
        )
        self.assertEqual(len(contract.rows), 10)
        self.assertIsInstance(contract.seed_normal_confirmations, MappingProxyType)
        self.assertEqual(tuple(contract.seed_normal_confirmations), (20260721, 20261721))
        with self.assertRaises(TypeError):
            contract.seed_normal_confirmations[20260721] = 1  # type: ignore[index]

        state = initial_governance_state_antigravity()
        self.assertEqual(state["label_query_count"], 0)
        self.assertEqual(state["official_evaluate_count"], 0)
        self.assertIsNone(state["artifact_map_sha256"])

        reordered = {20261721: 0, 20260721: 0}
        with self.assertRaises(ValueError):
            replace(contract, seed_normal_confirmations=reordered)
        with self.assertRaises(ValueError):
            replace(contract, rows=rows[:9])
        with self.assertRaises(ValueError):
            replace(contract, label_query_count=True)
        with self.assertRaises(ValueError):
            replace(contract, official_evaluate_count=1)
        with self.assertRaises(ValueError):
            replace(contract, artifact_map_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
