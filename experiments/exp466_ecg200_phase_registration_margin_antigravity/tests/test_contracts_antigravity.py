from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
from types import MappingProxyType
import unittest


class Exp466ContractsTestAntigravity(unittest.TestCase):
    def phase_evidence_antigravity(self, **changes):
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.contracts_antigravity import (
            Exp466PhaseEvidenceAntigravity,
        )

        values = {
            "identity_error": 0.40,
            "undo_error": 0.15,
            "margin": 0.25,
            "fit_normal_max_margin": 0.10,
            "confirmed": True,
            "evidence_sha256": "a" * 64,
        }
        values.update(changes)
        return Exp466PhaseEvidenceAntigravity(**values)

    def fold_result_antigravity(self, *, seed=20260721, fold_index=0, **changes):
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.contracts_antigravity import (
            Exp466FoldResultAntigravity,
        )

        values = {
            "seed": seed,
            "fold_index": fold_index,
            "per_type_auc": (0.90, 0.91, 0.92, 0.93),
            "mean_auc": 0.915,
            "c0_gradual_auc": 0.88,
            "c0_period_auc": 0.89,
            "c9_gradual_auc": 0.91,
            "c9_period_auc": 0.93,
            "period_confirmations": 1,
            "all_scores_finite": True,
            "passed": True,
            "failure_reasons": (),
        }
        values.update(changes)
        return Exp466FoldResultAntigravity(**values)

    def valid_qualification_antigravity(self):
        from experiments.exp466_ecg200_phase_registration_margin_antigravity.contracts_antigravity import (
            Exp466QualificationAntigravity,
        )

        rows = tuple(
            self.fold_result_antigravity(seed=seed, fold_index=fold_index)
            for seed in (20260721, 20261721)
            for fold_index in range(5)
        )
        return Exp466QualificationAntigravity(
            fold_results=rows,
            normal_trial_counts=MappingProxyType({20260721: 69, 20261721: 69}),
            normal_confirmations=MappingProxyType({20260721: 0, 20261721: 0}),
            label_query_count=0,
            official_evaluate_count=0,
            official_inspect_count=0,
            artifact_map_sha256=None,
        )

    def test_phase_evidence_rejects_nonfinite_margin_and_wrong_sha_antigravity(self) -> None:
        """Break caught: non-finite phase evidence or a forged digest enters replay."""
        evidence = self.phase_evidence_antigravity()
        self.assertEqual(evidence.margin, 0.25)
        with self.assertRaises(FrozenInstanceError):
            evidence.margin = 0.20  # type: ignore[misc]
        for changes in (
            {"margin": float("nan")},
            {"margin": float("inf")},
            {"evidence_sha256": "A" * 64},
            {"evidence_sha256": "a" * 63},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.phase_evidence_antigravity(**changes)

    def test_fold_result_rejects_gate_shape_aliases_and_inconsistent_mean_antigravity(self) -> None:
        """Break caught: a forged fold row bypasses finite, count, or gate shape checks."""
        for changes in (
            {"seed": True},
            {"fold_index": 5},
            {"per_type_auc": [0.90, 0.91, 0.92, 0.93]},
            {"per_type_auc": (0.90, 0.91, float("nan"), 0.93)},
            {"mean_auc": 0.99},
            {"period_confirmations": 0},
            {"all_scores_finite": 1},
            {"passed": False, "failure_reasons": ()},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.fold_result_antigravity(**changes)

    def test_qualification_rejects_mutable_reordered_and_wrong_69_counts_antigravity(self) -> None:
        """Break caught: aggregate governance accepts mutable, reordered, or partial OOF truth."""
        valid = self.valid_qualification_antigravity()
        self.assertIsInstance(valid.normal_confirmations, MappingProxyType)
        self.assertEqual(tuple(valid.normal_confirmations), (20260721, 20261721))
        with self.assertRaisesRegex(ValueError, "immutable"):
            dataclasses.replace(valid, normal_confirmations={20260721: 0, 20261721: 0})
        with self.assertRaisesRegex(ValueError, "normal confirmation"):
            dataclasses.replace(
                valid,
                normal_confirmations=MappingProxyType({20261721: 0, 20260721: 0}),
            )
        with self.assertRaisesRegex(ValueError, "normal confirmation"):
            dataclasses.replace(
                valid,
                normal_confirmations=MappingProxyType({20260721: 1, 20261721: 0}),
            )
        with self.assertRaisesRegex(ValueError, "69"):
            dataclasses.replace(
                valid,
                normal_trial_counts=MappingProxyType({20260721: 68, 20261721: 69}),
            )
        with self.assertRaisesRegex(ValueError, "fold_results"):
            dataclasses.replace(valid, fold_results=valid.fold_results[:-1])

    def test_qualification_rejects_nonzero_official_or_label_counts_antigravity(self) -> None:
        """Break caught: Task 1 records an official command, label query, or result hash."""
        valid = self.valid_qualification_antigravity()
        for field_name in (
            "label_query_count",
            "official_evaluate_count",
            "official_inspect_count",
        ):
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                dataclasses.replace(valid, **{field_name: 1})
        with self.assertRaises(ValueError):
            dataclasses.replace(valid, artifact_map_sha256="b" * 64)


if __name__ == "__main__":
    unittest.main()
