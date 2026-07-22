import unittest

from experiments.exp328_large_fn_reachability.pipeline import classify_false_negative


class ReachabilityClassificationTests(unittest.TestCase):
    def test_review_lane_has_highest_priority(self):
        result = classify_false_negative(
            7, review={7}, primary_candidates={"a": {7}}, primary_orders={"a": [7]}, independent={}
        )
        self.assertEqual(result["category"], "already_in_review_lane")

    def test_two_independent_candidates_are_consensus_routing_miss(self):
        result = classify_false_negative(
            7, review=set(), primary_candidates={}, primary_orders={}, independent={"a": {7}, "b": {7}}
        )
        self.assertEqual(result["category"], "multi_source_candidate_not_routed")

    def test_one_candidate_is_consensus_gate_miss(self):
        result = classify_false_negative(
            7, review=set(), primary_candidates={"a": {7}}, primary_orders={"a": [7]}, independent={}
        )
        self.assertEqual(result["category"], "single_source_candidate")

    def test_ranked_below_threshold_is_threshold_miss(self):
        result = classify_false_negative(
            7, review=set(), primary_candidates={"a": set()}, primary_orders={"a": [1, 7]}, independent={}
        )
        self.assertEqual(result["category"], "top12_below_threshold")

    def test_outside_top64_is_representation_miss(self):
        result = classify_false_negative(
            100, review=set(), primary_candidates={"a": set()}, primary_orders={"a": list(range(64))}, independent={}
        )
        self.assertEqual(result["category"], "outside_audited_top64")


if __name__ == "__main__":
    unittest.main()
