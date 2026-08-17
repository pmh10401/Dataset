from __future__ import annotations

import unittest


class RouterAntigravityTest(unittest.TestCase):
    def test_thresholds_and_equal_outcome_tie_keep_the_baseline_antigravity(self) -> None:
        """Break caught: a threshold direction or the deterministic baseline tie break changes."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import ConfusionAntigravity, DatasetRoutingFeaturesAntigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import applies_antigravity, fit_family_rule_antigravity, threshold_rules_antigravity

        rules = threshold_rules_antigravity("target_length", (3, 5))
        self.assertEqual(tuple((rule.threshold_twice, rule.candidate_when) for rule in rules), ((None, "never"), (None, "always"), (8, "lte"), (8, "gt")))
        self.assertTrue(applies_antigravity(rules[2], 4))
        self.assertFalse(applies_antigravity(rules[2], 5))
        features = tuple(DatasetRoutingFeaturesAntigravity(f"ECG5000__{index}", "ECG5000", 10 + index, 1, 2, 1, 2, 1, 0, 500000) for index in range(2))
        outcomes = {item.dataset: (ConfusionAntigravity(10, 1, 2, 100), ConfusionAntigravity(10, 1, 2, 100)) for item in features}
        rule = fit_family_rule_antigravity(features, outcomes)
        self.assertEqual((rule.feature_name, rule.threshold_twice, rule.candidate_when), (None, None, "never"))
