from __future__ import annotations

import unittest


class FamilyRouterAntigravityTests(unittest.TestCase):
    """Tests for the frozen, feature-only Task 3 routing rule selection."""

    @staticmethod
    def _feature_antigravity(dataset: str, family: str, target_length: int):
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import DatasetRoutingFeaturesAntigravity

        return DatasetRoutingFeaturesAntigravity(
            dataset=dataset,
            family=family,
            target_length=target_length,
            baseline_count=target_length // 10,
            candidate_count=target_length // 10 + 1,
            overlap_count=target_length // 20,
            union_count=target_length // 8 + 1,
            added_count=1,
            removed_count=0,
            overlap_ppm=500_000,
        )

    @staticmethod
    def _outcome_antigravity(baseline_fn: int, candidate_fn: int):
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.contracts_antigravity import ConfusionAntigravity

        return (
            ConfusionAntigravity(tp=10, fp=1, fn=baseline_fn, tn=100),
            ConfusionAntigravity(tp=10, fp=1, fn=candidate_fn, tn=100),
        )

    def test_threshold_rules_use_integer_midpoints_and_both_directions_antigravity(self) -> None:
        """Break caught: a midpoint boundary or one threshold direction is omitted."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import (
            applies_antigravity,
            threshold_rules_antigravity,
        )

        rules = threshold_rules_antigravity("target_length", (3, 5, 10))
        self.assertEqual(
            tuple((rule.feature_name, rule.threshold_twice, rule.candidate_when) for rule in rules),
            ((None, None, "never"), (None, None, "always"), ("target_length", 8, "lte"), ("target_length", 8, "gt"), ("target_length", 15, "lte"), ("target_length", 15, "gt")),
        )
        self.assertTrue(applies_antigravity(rules[2], 4))
        self.assertFalse(applies_antigravity(rules[2], 5))
        self.assertFalse(applies_antigravity(rules[3], 4))
        self.assertTrue(applies_antigravity(rules[3], 5))

    def test_exact_outcome_tie_prefers_baseline_route_antigravity(self) -> None:
        """Break caught: an equal candidate route beats the baseline despite the tie rule."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import fit_family_rule_antigravity

        features = (
            self._feature_antigravity("ECG5000__a", "ECG5000", 10),
            self._feature_antigravity("ECG5000__b", "ECG5000", 20),
        )
        outcomes = {row.dataset: self._outcome_antigravity(2, 2) for row in features}
        rule = fit_family_rule_antigravity(features, outcomes)
        self.assertEqual((rule.feature_name, rule.threshold_twice, rule.candidate_when), (None, None, "never"))

    def test_held_out_outcome_never_changes_its_fold_rule_or_selection_antigravity(self) -> None:
        """Break caught: the held-out confusion leaks into its own LODO rule selection."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        ecg = tuple(self._feature_antigravity(f"ECG5000__{index}", "ECG5000", 10 + index * 10) for index in range(5))
        chlorine = tuple(self._feature_antigravity(f"ChlorineConcentration__{index}", "ChlorineConcentration", 100 + index * 10) for index in range(3))
        features = ecg + chlorine
        outcomes = {row.dataset: self._outcome_antigravity(5, 1) for row in ecg}
        outcomes.update({row.dataset: self._outcome_antigravity(1, 5) for row in chlorine})
        original = {decision.dataset: decision for decision in leave_one_dataset_out_antigravity(features, outcomes)}
        held_out = features[2].dataset
        mutated = dict(outcomes)
        mutated[held_out] = self._outcome_antigravity(9_999, 0)
        changed = {decision.dataset: decision for decision in leave_one_dataset_out_antigravity(features, mutated)}
        self.assertEqual(changed[held_out].rule, original[held_out].rule)
        self.assertEqual(changed[held_out].selected_route, original[held_out].selected_route)

    def test_training_order_cannot_change_rule_antigravity(self) -> None:
        """Break caught: rule selection depends on input ordering rather than its aggregate score."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import fit_family_rule_antigravity

        features = tuple(self._feature_antigravity(f"ECG5000__{index}", "ECG5000", value) for index, value in enumerate((50, 10, 80, 30)))
        outcomes = {
            features[0].dataset: self._outcome_antigravity(2, 3),
            features[1].dataset: self._outcome_antigravity(4, 1),
            features[2].dataset: self._outcome_antigravity(2, 3),
            features[3].dataset: self._outcome_antigravity(4, 1),
        }
        self.assertEqual(
            fit_family_rule_antigravity(features, outcomes),
            fit_family_rule_antigravity(tuple(reversed(features)), outcomes),
        )

    def test_families_are_separate_and_eight_folds_keep_input_order_antigravity(self) -> None:
        """Break caught: a family leaks into another family's fit or folds lose their identity/order."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        ecg = tuple(self._feature_antigravity(f"ECG5000__{index}", "ECG5000", 100 + index) for index in range(5))
        chlorine = tuple(self._feature_antigravity(f"ChlorineConcentration__{index}", "ChlorineConcentration", 200 + index) for index in range(3))
        features = ecg + chlorine
        outcomes = {row.dataset: self._outcome_antigravity(5, 1) for row in ecg}
        outcomes.update({row.dataset: self._outcome_antigravity(1, 5) for row in chlorine})
        decisions = leave_one_dataset_out_antigravity(features, outcomes)
        self.assertEqual(tuple(decision.dataset for decision in decisions), tuple(row.dataset for row in features))
        self.assertEqual(tuple(decision.family for decision in decisions), ("ECG5000",) * 5 + ("ChlorineConcentration",) * 3)
        self.assertTrue(all(decision.method == "hydra_local_gap" for decision in decisions[:5]))
        self.assertTrue(all(decision.method == "multirocket_local_gap" for decision in decisions[5:]))
        self.assertTrue(all(decision.selected_route == "candidate" for decision in decisions[:5]))
        self.assertTrue(all(decision.selected_route == "baseline" for decision in decisions[5:]))
        reordered = chlorine + ecg
        reordered_decisions = leave_one_dataset_out_antigravity(reordered, outcomes)
        self.assertEqual(tuple(decision.dataset for decision in reordered_decisions), tuple(row.dataset for row in reordered))
        self.assertEqual(
            {decision.dataset: decision.selected_route for decision in reordered_decisions},
            {decision.dataset: decision.selected_route for decision in decisions},
        )


if __name__ == "__main__":
    unittest.main()
