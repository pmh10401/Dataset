from __future__ import annotations

import unittest
from experiments.exp161.alternative_source_audit_core import (
    is_target,
    detect_computed_exp99,
    detect_computed_exp101,
    detect_computed_exp46,
    is_policy_eligible,
    parse_indices,
    calculate_legacy_index_overlap,
)


class TestAlternativeSourceAuditCore(unittest.TestCase):

    def test_target_selection(self):
        # 1. Exp160 target selection correctness (n_train >= 5, candidate_count == 0)
        self.assertTrue(is_target({"n_train": 5, "candidate_count": 0}))
        self.assertTrue(is_target({"n_train": 10, "candidate_count": 0}))
        self.assertTrue(is_target({"n_train": "100", "candidate_count": "0"}))

        # 2. n_train < 5 exclusion
        self.assertFalse(is_target({"n_train": 4, "candidate_count": 0}))
        self.assertFalse(is_target({"n_train": 1, "candidate_count": 0}))
        self.assertFalse(is_target({"n_train": 0, "candidate_count": 0}))
        self.assertFalse(is_target({"n_train": -1, "candidate_count": 0}))

        # candidate_count > 0 should not be a target
        self.assertFalse(is_target({"n_train": 5, "candidate_count": 1}))
        self.assertFalse(is_target({"n_train": 10, "candidate_count": 5}))

    def test_exp99_computation_status(self):
        # 3. Exp99 passthrough vs actual spectral calculation distinction
        computed_rows = [
            {"dataset_name": "DS1", "spectral_target": "1", "research_target": "0", "selector_reason": "spectral/derivative robust score candidate"},
            {"dataset_name": "DS2", "spectral_target": "0", "research_target": "1", "selector_reason": "spectral/derivative robust score candidate"},
            {"dataset_name": "DS3", "spectral_target": "0", "research_target": "0", "selector_reason": "control: Exp93 operating default"},  # but reason is not non-target passthrough
        ]
        passthrough_rows = [
            {"dataset_name": "DS4", "spectral_target": "0", "research_target": "0", "selector_reason": "non-target passthrough"},
            {"dataset_name": "DS5", "spectral_target": "", "research_target": "", "selector_reason": "non-target passthrough"},
        ]

        computed = detect_computed_exp99(computed_rows + passthrough_rows)
        self.assertIn("DS1", computed)
        self.assertIn("DS2", computed)
        self.assertNotIn("DS3", computed)
        self.assertNotIn("DS4", computed)
        self.assertNotIn("DS5", computed)

    def test_exp101_computation_status(self):
        # 4. Exp101 passthrough vs actual shapelet calculation distinction
        computed_rows = [
            {"dataset_name": "DS1", "shapelet_target": "1", "research_target": "0", "selector_reason": "shapelet normal prototype robust score candidate"},
            {"dataset_name": "DS2", "shapelet_target": "0", "research_target": "1", "selector_reason": "shapelet normal prototype robust score candidate"},
            {"dataset_name": "DS3", "shapelet_target": "0", "research_target": "0", "selector_reason": "control: Exp93 operating default"},
        ]
        passthrough_rows = [
            {"dataset_name": "DS4", "shapelet_target": "0", "research_target": "0", "selector_reason": "non-target passthrough"},
        ]

        computed = detect_computed_exp101(computed_rows + passthrough_rows)
        self.assertIn("DS1", computed)
        self.assertIn("DS2", computed)
        self.assertNotIn("DS3", computed)
        self.assertNotIn("DS4", computed)

    def test_exp46_computation_status_and_partial_coverage(self):
        # 5. Exp46 13 partial coverage indication
        rows = [
            {"dataset_name": "ScreenType_normal_1"},  # in 13
            {"dataset_name": "Computers_normal_1"},   # in 13
            {"dataset_name": "SomeOther_normal_99"},  # not in 13
        ]
        computed = detect_computed_exp46(rows)
        self.assertIn("ScreenType_normal_1", computed)
        self.assertIn("Computers_normal_1", computed)
        self.assertNotIn("SomeOther_normal_99", computed)

    def test_policy_eligibility(self):
        # 6. family/research target provenance is policy eligible false
        # Exp99/101 must always return policy_eligible = False
        exp99_computed = {"DS_spectral_target", "DS_research_target"}
        exp101_computed = {"DS_shapelet_target", "DS_research_target"}
        exp46_computed = {"Computers_normal_1"}

        self.assertFalse(is_policy_eligible("spectral_derivative", "DS_spectral_target", exp99_computed, exp101_computed, exp46_computed))
        self.assertFalse(is_policy_eligible("spectral_derivative", "DS_research_target", exp99_computed, exp101_computed, exp46_computed))
        self.assertFalse(is_policy_eligible("shapelet_normal_prototype", "DS_shapelet_target", exp99_computed, exp101_computed, exp46_computed))
        self.assertFalse(is_policy_eligible("shapelet_normal_prototype", "DS_research_target", exp99_computed, exp101_computed, exp46_computed))

        # Exp46 was selected by family/difficulty, so its historical artifact is
        # research coverage evidence only, never policy-eligible.
        self.assertFalse(is_policy_eligible("interval_quantile", "Computers_normal_1", exp99_computed, exp101_computed, exp46_computed))
        # Non-computed in Exp46 should be False
        self.assertFalse(is_policy_eligible("interval_quantile", "Computers_normal_3", exp99_computed, exp101_computed, exp46_computed))

    def test_metric_independence(self):
        # 7. metric values changed should NOT affect target selection, computed detection, or policy eligibility
        row = {"n_train": 5, "candidate_count": 0, "tp": 10, "fp": 2, "fn": 0, "f1": 0.9, "auc_roc": 0.95, "oracle_f1": 1.0}
        self.assertTrue(is_target(row))

        # Modifying metrics
        row_bad_metrics = {"n_train": 5, "candidate_count": 0, "tp": 0, "fp": 20, "fn": 10, "f1": 0.0, "auc_roc": 0.1, "oracle_f1": 0.1}
        self.assertTrue(is_target(row_bad_metrics))

        # computed check
        row_exp99 = {"dataset_name": "DS1", "spectral_target": "1", "f1": "0.99", "tp": "5"}
        computed1 = detect_computed_exp99([row_exp99])
        row_exp99_bad = {"dataset_name": "DS1", "spectral_target": "1", "f1": "0.0", "tp": "0"}
        computed2 = detect_computed_exp99([row_exp99_bad])
        self.assertEqual(computed1, computed2)

    def test_legacy_index_overlap_and_passthrough_evidence(self):
        # 8. selected_indices not used as passthrough source evidence (return None if not computed or not research_target)
        exp_rows = [
            {"dataset_name": "DS1", "selector_name": "research_spectral_q98_cap2", "selected_indices": "1 2 3", "research_target": "1"},
            {"dataset_name": "DS1", "selector_name": "baseline_exp93_operating", "selected_indices": "2 3 4", "research_target": "1"},
        ]

        # Case A: Computed & research_target == 1 -> returns overlap
        overlap = calculate_legacy_index_overlap("spectral_derivative", "DS1", exp_rows, is_computed=True)
        self.assertIsNotNone(overlap)
        self.assertEqual(overlap["overlap_count"], 2)
        self.assertEqual(overlap["overlap_indices"], [2, 3])

        # Case B: Not computed -> returns None
        overlap_not_computed = calculate_legacy_index_overlap("spectral_derivative", "DS1", exp_rows, is_computed=False)
        self.assertIsNone(overlap_not_computed)

        # Case C: computed but research_target == 0
        exp_rows_no_research = [
            {"dataset_name": "DS1", "selector_name": "research_spectral_q98_cap2", "selected_indices": "1 2 3", "research_target": "0"},
            {"dataset_name": "DS1", "selector_name": "baseline_exp93_operating", "selected_indices": "2 3 4", "research_target": "0"},
        ]
        overlap_no_research = calculate_legacy_index_overlap("spectral_derivative", "DS1", exp_rows_no_research, is_computed=True)
        self.assertIsNone(overlap_no_research)

    def test_parse_indices(self):
        self.assertEqual(parse_indices("1,2,3"), {1, 2, 3})
        self.assertEqual(parse_indices("1 2 3"), {1, 2, 3})
        self.assertEqual(parse_indices("1;2;3"), {1, 2, 3})
        self.assertEqual(parse_indices(""), set())
        self.assertEqual(parse_indices(None), set())


if __name__ == "__main__":
    unittest.main()
