from __future__ import annotations

import unittest

from experiments.exp392_label_guided_discovery.discovery_core import (
    CombinationRule,
    benjamini_hochberg,
    candidate_indices_for_rule,
    enumerate_combination_rules,
    evaluate_candidate_set,
)


class CombinationEnumerationTests(unittest.TestCase):
    def test_enumerates_every_pair_and_triple_without_duplicates(self):
        rules = enumerate_combination_rules(("a", "b", "c"), alpha=0.005)
        names = [rule.rule_id for rule in rules]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("intersection__a__b__alpha_0p005", names)
        self.assertIn("union__a__b__alpha_0p005", names)
        self.assertIn("atleast2__a__b__c__alpha_0p005", names)
        self.assertEqual(len(rules), 9)

    def test_candidate_rule_never_reintroduces_excluded_indices(self):
        rule = CombinationRule("intersection", ("a", "b"), 0.005, 2)

        result = candidate_indices_for_rule(
            rule,
            {"a": {1, 2}, "b": {2, 3}},
            excluded={2},
            n_test=5,
        )

        self.assertEqual(result, set())

    def test_atleast_two_rule_counts_votes(self):
        rule = CombinationRule("atleast2", ("a", "b", "c"), 0.005, 2)

        result = candidate_indices_for_rule(
            rule,
            {"a": {1, 2}, "b": {2, 3}, "c": {2, 3, 4}},
            excluded=set(),
            n_test=5,
        )

        self.assertEqual(result, {2, 3})

    def test_rejects_out_of_range_source_index(self):
        rule = CombinationRule("union", ("a", "b"), 0.005, 1)

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            candidate_indices_for_rule(
                rule,
                {"a": {1, 5}, "b": set()},
                excluded=set(),
                n_test=5,
            )


class CandidateEvaluationTests(unittest.TestCase):
    def test_reports_fn_recovery_fp_and_review_overlap(self):
        result = evaluate_candidate_set(
            candidate_indices={1, 2, 4},
            hard_indices={0},
            review_indices={1, 2},
            labels=("0", "1", "0", "1", "1"),
        )

        self.assertEqual(result["baseline_fn"], 3)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["recovered_fn"], 2)
        self.assertEqual(result["added_fp"], 1)
        self.assertEqual(result["candidate_precision"], 2 / 3)
        self.assertEqual(result["review_overlap_count"], 2)
        self.assertEqual(result["review_recovered_fn"], 1)
        self.assertEqual(result["review_overlap_fp"], 1)
        self.assertEqual(result["novel_recovered_fn"], 1)


class MultipleComparisonTests(unittest.TestCase):
    def test_benjamini_hochberg_preserves_order_and_is_monotone(self):
        adjusted = benjamini_hochberg([0.04, 0.001, 0.03, 0.2])

        self.assertEqual(len(adjusted), 4)
        self.assertAlmostEqual(adjusted[0], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[1], 0.004)
        self.assertAlmostEqual(adjusted[2], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[3], 0.2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))


if __name__ == "__main__":
    unittest.main()
