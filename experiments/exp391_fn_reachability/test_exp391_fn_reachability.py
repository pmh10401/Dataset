from __future__ import annotations

import unittest

from experiments.exp391_fn_reachability.fn_reachability_core import (
    build_vote_candidates,
    classify_baseline,
    evaluate_recovery,
)


class BaselineClassificationTests(unittest.TestCase):
    def test_separates_no_hard_wrong_hard_and_detected_datasets(self):
        labels = ["0", "1", "0", "1"]

        self.assertEqual(classify_baseline([], labels), "zero_no_hard")
        self.assertEqual(classify_baseline([0, 2], labels), "zero_wrong_hard")
        self.assertEqual(classify_baseline([1], labels), "hard_has_tp")


class CandidateVoteTests(unittest.TestCase):
    def test_excludes_existing_hard_and_exact_normal_indices(self):
        evidence = {
            "frequency": {"available": True, "indices": [0, 1, 3]},
            "ewma": {"available": True, "indices": [1, 2, 3]},
            "temporal": {"available": False, "indices": []},
        }

        candidates = build_vote_candidates(
            evidence,
            vote_threshold=2,
            n_test=5,
            excluded_indices={1, 2},
        )

        self.assertEqual(candidates, {3})

    def test_rejects_out_of_range_evidence_indices(self):
        evidence = {"frequency": {"available": True, "indices": [4]}}

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            build_vote_candidates(
                evidence,
                vote_threshold=1,
                n_test=4,
                excluded_indices=set(),
            )


class RecoveryEvaluationTests(unittest.TestCase):
    def test_reports_recovered_fn_added_fp_and_review_overlap(self):
        result = evaluate_recovery(
            candidate_indices={1, 2, 4},
            hard_indices={0},
            review_indices={1, 3},
            labels=["0", "1", "0", "1", "1"],
        )

        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["recovered_fn"], 2)
        self.assertEqual(result["added_fp"], 1)
        self.assertEqual(result["candidate_precision"], 2 / 3)
        self.assertEqual(result["review_overlap_count"], 1)
        self.assertEqual(result["review_recovered_fn"], 1)
        self.assertEqual(result["novel_recovered_fn"], 1)


if __name__ == "__main__":
    unittest.main()
