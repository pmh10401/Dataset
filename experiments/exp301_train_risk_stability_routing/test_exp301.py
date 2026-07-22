from __future__ import annotations

import unittest

import numpy as np

from experiments.exp301_train_risk_stability_routing.core import (
    bootstrap_consensus_stability,
    build_routing_row,
    consensus_train_flags,
    empirical_upper_pvalues,
    evaluate_predictions,
    loo_upper_pvalues,
    summarize_evaluated,
    wilson_upper,
)


class TrainRiskTests(unittest.TestCase):
    def test_leave_one_out_pvalues_count_ties_conservatively(self):
        actual = loo_upper_pvalues(np.array([1.0, 2.0, 2.0, 3.0]))
        np.testing.assert_allclose(actual, [1.0, 0.75, 0.75, 0.25])

    def test_empirical_pvalues_use_plus_one_correction(self):
        actual = empirical_upper_pvalues(np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(actual, [1.0, 0.75, 0.25])

    def test_three_source_train_consensus_requires_both_multirocket_seeds(self):
        multi = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 2.0, 3.0]])
        structural = np.array([1.0, 2.0, 3.0, 4.0])
        change = np.array([1.0, 2.0, 3.0, 4.0])
        flags = consensus_train_flags(multi, structural, change, alpha=0.50)
        np.testing.assert_array_equal(flags["multirocket"], [False, False, False, True])
        np.testing.assert_array_equal(flags["three_of_three"], [False, False, False, True])

    def test_wilson_upper_is_conservative(self):
        self.assertGreater(wilson_upper(0, 100), 0.0)
        self.assertLess(wilson_upper(0, 100), 0.05)

    def test_bootstrap_stability_is_deterministic_and_uses_three_sources(self):
        kwargs = {
            "dataset": "Demo",
            "multirocket_train_by_seed": np.array([[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]]),
            "multirocket_test_by_seed": np.array([[10.0, 0.0], [10.0, 0.0]]),
            "structural_train": np.array([0.0, 0.1, 0.2, 0.3]),
            "structural_test": np.array([10.0, 0.0]),
            "change_point_train": np.array([0.0, 0.1, 0.2, 0.3]),
            "change_point_test": np.array([10.0, 0.0]),
            "candidate_indices": [0, 1],
            "alpha": 0.25,
            "replicates": 8,
            "seed": 7,
        }
        first = bootstrap_consensus_stability(**kwargs)
        second = bootstrap_consensus_stability(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 1.0)
        self.assertEqual(first[1], 0.0)


class RoutingTests(unittest.TestCase):
    def test_routes_stable_candidates_into_mutually_exclusive_shadow_lanes(self):
        row = build_routing_row(
            dataset="Demo",
            n_train=120,
            n_test=8,
            current_hard_indices=[7],
            general_candidates=[1, 2, 7],
            automatic_candidates=[2, 3, 7],
            priority_candidates=[2, 3, 4, 7],
            automatic_stability={2: 0.75, 3: 0.875},
            priority_stability={2: 0.90, 3: 0.875, 4: 0.50},
            automatic_risk_contract_pass=True,
        )
        self.assertEqual(row["automatic_qualification_indices"], [3])
        self.assertEqual(row["priority_review_indices"], [2, 4])
        self.assertEqual(row["general_review_indices"], [1])
        self.assertFalse(row["autonomous_policy_changed"])
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        lanes = [set(row[name]) for name in ("automatic_qualification_indices", "priority_review_indices", "general_review_indices")]
        self.assertFalse((lanes[0] & lanes[1]) | (lanes[0] & lanes[2]) | (lanes[1] & lanes[2]))

    def test_failed_train_risk_contract_blocks_automatic_qualification(self):
        row = build_routing_row(
            dataset="Demo", n_train=120, n_test=5, current_hard_indices=[],
            general_candidates=[], automatic_candidates=[1], priority_candidates=[1],
            automatic_stability={1: 1.0}, priority_stability={1: 1.0},
            automatic_risk_contract_pass=False,
        )
        self.assertEqual(row["automatic_qualification_indices"], [])
        self.assertEqual(row["priority_review_indices"], [1])

    def test_evaluation_is_separate_and_summary_keeps_hard_unchanged(self):
        prediction = build_routing_row(
            dataset="Demo", n_train=120, n_test=6, current_hard_indices=[0],
            general_candidates=[2], automatic_candidates=[1], priority_candidates=[1, 3],
            automatic_stability={1: 1.0}, priority_stability={1: 1.0, 3: 1.0},
            automatic_risk_contract_pass=True,
        )
        evaluated = evaluate_predictions([prediction], {"Demo": np.array([1, 1, 0, 1, 0, 0])})
        row = evaluated[0]
        self.assertEqual((row["baseline_hard_tp"], row["baseline_hard_fp"]), (1, 0))
        self.assertEqual((row["automatic_qualification_tp"], row["automatic_qualification_fp"]), (1, 0))
        self.assertEqual((row["priority_review_tp"], row["priority_review_fp"]), (1, 0))
        summary = summarize_evaluated(evaluated)[0]
        self.assertEqual(summary["datasets"], 1)
        self.assertEqual(summary["baseline_hard_count"], 1)
        self.assertFalse(summary["autonomous_policy_changed"])


if __name__ == "__main__":
    unittest.main()
