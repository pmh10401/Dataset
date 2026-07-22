from __future__ import annotations

import unittest

from experiments.exp274_275_large_fn_recovery.policy import build_prediction_row


class RecoveryPolicyTests(unittest.TestCase):
    def _base(self, n_train: int = 120) -> dict[str, object]:
        return {
            "dataset": "fixture",
            "n_train": n_train,
            "n_test": 8,
            "hard_indices": [7],
            "shadow_indices": [],
            "priority_indices": [],
            "standard_indices": [6],
            "recovery_shadow_indices": [1, 2, 3],
            "recovery_priority_indices": [4],
            "data_quality_review_indices": [],
            "one_source_sensitivity_indices": [0, 5],
            "support_by_index": {
                "0": ["rff_forecast+tcn_seed_intersection+tcn_seed_median"],
                "1": ["a", "b", "c"],
                "2": ["a", "b", "c"],
                "3": ["a", "b", "c"],
                "5": ["signature"],
            },
        }

    def test_frozen_variants_follow_preregistered_rules(self) -> None:
        row = build_prediction_row(
            self._base(), wavelet={0, 1, 2, 5}, matrix_profile={0, 2, 3, 5},
            input_sha256={"a": "b"},
        )
        self.assertEqual(row["forecast_wavelet_matrix_indices"], [0])
        self.assertEqual(row["any_source_wavelet_matrix_indices"], [0, 5])
        self.assertEqual(row["shadow_either_structure_indices"], [1, 2, 3])
        self.assertEqual(row["shadow_both_structures_indices"], [2])

    def test_expansion_abstains_below_train_minimum(self) -> None:
        row = build_prediction_row(
            self._base(n_train=99), wavelet={0, 1}, matrix_profile={0, 1},
            input_sha256={"a": "b"},
        )
        self.assertEqual(row["forecast_wavelet_matrix_indices"], [])
        self.assertEqual(row["any_source_wavelet_matrix_indices"], [])
        self.assertTrue(row["large_train_abstained"])

    def test_prediction_has_no_label_or_metric_fields(self) -> None:
        row = build_prediction_row(
            self._base(), wavelet=set(), matrix_profile=set(), input_sha256={"a": "b"},
        )
        forbidden = {"y_test", "true_labels", "tp", "fp", "f1", "precision", "recall"}
        self.assertTrue(forbidden.isdisjoint(row))
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertFalse(row["hard_policy_changed"])


if __name__ == "__main__":
    unittest.main()

