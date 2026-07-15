from __future__ import annotations

import unittest
import numpy as np

from experiments.exp163.alternative_tail_core import (
    calculate_quantile_threshold,
    find_tail_candidates,
    compute_agreement,
    integrate_lanes,
    is_candidate_zero_target,
    validate_lane_contract,
    build_prediction_row,
    evaluate_prediction_rows,
)


class TestExp163AlternativeTailReview(unittest.TestCase):

    def test_quantile_threshold_higher(self):
        # np.quantile(..., method="higher") 테스트
        # 1-alpha 분위수. alpha=0.05 이면 q=0.95
        # train_scores 에 finite 값과 infinite(nan, inf) 값이 섞여 있을 때 finite만 골라내는지 확인
        train_scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan, np.inf, -np.inf])
        # finite: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # q=0.95 -> index는 9 * 0.95 = 8.55 -> higher method는 index 9의 값인 10.0을 선택해야 함.
        # q=0.90 -> index는 9 * 0.90 = 8.1 -> higher method는 index 9의 값인 10.0을 선택해야 함.
        # q=0.80 -> index는 9 * 0.80 = 7.2 -> higher method는 index 8의 값인 9.0을 선택해야 함.
        threshold_005 = calculate_quantile_threshold(train_scores, 0.05)
        self.assertEqual(threshold_005, 10.0)

        threshold_020 = calculate_quantile_threshold(train_scores, 0.20)
        self.assertEqual(threshold_020, 9.0)

    def test_strict_greater_than_ties(self):
        # test_score > threshold 여야만 함. threshold 와 동점인 경우는 포함되면 안 됨.
        test_scores = np.array([5.0, 6.0, 4.0, 7.0, 6.0])
        # threshold 가 6.0 일 때, 7.0 (index 3)만 후보가 되어야 함. 6.0은 > 가 아니므로 탈락.
        candidates = find_tail_candidates(test_scores, 6.0)
        np.testing.assert_array_equal(candidates, np.array([3]))

    def test_agreement_2of3_and_3of3(self):
        # spectral, shapelet, interval indices 가 주어졌을 때 합의 집합
        spectral = {0, 1, 2}
        shapelet = {1, 2, 3}
        interval = {2, 3, 4}

        # 2of3: 적어도 2개 이상에 포함된 것 -> {1, 2, 3}
        ag_2of3 = compute_agreement(spectral, shapelet, interval, "tail_rank_2of3")
        self.assertEqual(ag_2of3, {1, 2, 3})

        # 3of3: 3개 모두에 포함된 것 -> {2}
        ag_3of3 = compute_agreement(spectral, shapelet, interval, "tail_rank_3of3")
        self.assertEqual(ag_3of3, {2})

        # Invalid variant type
        with self.assertRaises(ValueError):
            compute_agreement(spectral, shapelet, interval, "invalid_variant")

    def test_n_train_abstention(self):
        self.assertFalse(is_candidate_zero_target(n_train=4, baseline_candidate_count=0))
        self.assertTrue(is_candidate_zero_target(n_train=5, baseline_candidate_count=0))
        self.assertFalse(is_candidate_zero_target(n_train=5, baseline_candidate_count=1))

    def test_lane_integration_and_disjointness(self):
        # Hard/Priority baseline exact preservation
        # 기존 Exp160의 Hard, Priority lane 은 그대로 보존.
        # standard_review = (base_standard | new_agreement) - hard - priority
        hard = {1, 2}
        priority = {3}
        base_standard = {4}
        new_agreement = {2, 3, 4, 5, 6}

        new_hard, new_priority, new_standard = integrate_lanes(
            hard, priority, base_standard, new_agreement
        )

        self.assertEqual(new_hard, {1, 2})
        self.assertEqual(new_priority, {3})
        # (4 | 2, 3, 4, 5, 6) - 1, 2 - 3 => {2, 3, 4, 5, 6} - {1, 2, 3} => {4, 5, 6}
        self.assertEqual(new_standard, {4, 5, 6})

        # 상호 배타성 검증
        self.assertTrue(new_hard.isdisjoint(new_priority))
        self.assertTrue(new_hard.isdisjoint(new_standard))
        self.assertTrue(new_priority.isdisjoint(new_standard))

        candidate = new_hard | new_priority | new_standard
        validate_lane_contract(
            hard=new_hard,
            standard=new_standard,
            priority=new_priority,
            candidate=candidate,
            n_test=10,
        )

    def test_lane_contract_rejects_overlap_union_and_range_errors(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_lane_contract(
                hard={1}, standard={1}, priority=set(), candidate={1}, n_test=3
            )
        with self.assertRaisesRegex(ValueError, "union"):
            validate_lane_contract(
                hard={1}, standard=set(), priority=set(), candidate={1, 2}, n_test=3
            )
        with self.assertRaisesRegex(ValueError, "range"):
            validate_lane_contract(
                hard={3}, standard=set(), priority=set(), candidate={3}, n_test=3
            )

    def test_find_tail_candidates_excludes_non_finite_test_scores(self):
        test_scores = np.array([np.nan, np.inf, -np.inf, 4.0, 5.0])
        np.testing.assert_array_equal(find_tail_candidates(test_scores, 4.0), np.array([4]))

    def test_prediction_row_adds_only_standard_review_with_source_evidence(self):
        baseline = {
            "dataset_name": "d",
            "alpha": 0.05,
            "n_train": 5,
            "n_test": 3,
            "candidate_count": 0,
            "candidate_indices": "",
            "hard_indices": "",
            "standard_review_indices": "",
            "priority_review_indices": "",
            "abstained": 0,
        }
        scores = {
            "spectral_train": np.arange(5.0),
            "spectral_test": np.array([5.0, 6.0, 0.0]),
            "shapelet_train": np.arange(5.0),
            "shapelet_test": np.array([0.0, 6.0, 0.0]),
            "interval_train": np.arange(5.0),
            "interval_test": np.array([0.0, 6.0, 7.0]),
        }
        row = build_prediction_row(
            baseline_row=baseline,
            scores=scores,
            alpha=0.05,
            variant="tail_rank_2of3",
        )
        self.assertEqual(row["hard_indices"], "")
        self.assertEqual(row["priority_review_indices"], "")
        self.assertEqual(row["standard_review_indices"], "1")
        self.assertEqual(row["added_standard_indices"], "1")
        self.assertEqual(row["source_candidate_indices"]["spectral"], [0, 1])
        self.assertEqual(row["source_candidate_indices"]["shapelet"], [1])
        self.assertEqual(row["source_candidate_indices"]["interval"], [1, 2])
        self.assertEqual(row["source_thresholds"], {"spectral": 4.0, "shapelet": 4.0, "interval": 4.0})
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_n_train_below_five_preserves_frozen_baseline_lanes(self):
        baseline = {
            "dataset_name": "d",
            "alpha": 0.01,
            "n_train": 4,
            "n_test": 3,
            "candidate_count": 1,
            "candidate_indices": "1",
            "hard_indices": "1",
            "standard_review_indices": "",
            "priority_review_indices": "",
            "abstained": 1,
        }
        scores = {
            f"{source}_{split}": np.zeros(4 if split == "train" else 3)
            for source in ("spectral", "shapelet", "interval")
            for split in ("train", "test")
        }
        row = build_prediction_row(
            baseline_row=baseline,
            scores=scores,
            alpha=0.01,
            variant="tail_rank_2of3",
        )
        self.assertEqual(row["hard_indices"], "1")
        self.assertEqual(row["candidate_indices"], "1")
        self.assertEqual(row["candidate_zero_target"], 0)
        self.assertEqual(row["added_standard_count"], 0)

    def test_evaluation_is_separate_and_computes_added_review_metrics(self):
        prediction = {
            "dataset_name": "d",
            "alpha": 0.01,
            "variant": "tail_rank_2of3",
            "n_test": 3,
            "hard_indices": "0",
            "standard_review_indices": "1",
            "priority_review_indices": "",
            "candidate_indices": "0 1",
            "added_standard_indices": "1",
            "candidate_count": 2,
            "hard_count": 1,
            "standard_review_count": 1,
            "priority_review_count": 0,
            "abstained": 0,
        }
        evaluated = evaluate_prediction_rows(
            [prediction], label_loader=lambda _: np.array([0, 1, 0], dtype=int)
        )
        self.assertEqual(len(evaluated), 1)
        row = evaluated[0]
        self.assertEqual(row["hard_tp"], 0)
        self.assertEqual(row["hard_fp"], 1)
        self.assertEqual(row["added_standard_tp"], 1)
        self.assertEqual(row["added_standard_fp"], 0)
        self.assertEqual(row["combined_tp"], 1)
        self.assertEqual(row["combined_fp"], 1)


if __name__ == "__main__":
    unittest.main()
