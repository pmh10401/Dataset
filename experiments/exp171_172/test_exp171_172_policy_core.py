from __future__ import annotations

import math
import random
import unittest
from statistics import NormalDist

from experiments.exp169_170.pool_core import ALPHAS, POLICIES, SOURCE_CONFIGS
from experiments.exp171_172.policy_core import (
    expected_policy_keys,
    select_train_risk_policy,
    simultaneous_wilson_upper,
)


def make_rows(*, datasets: int = 2, trials: int = 3863):
    global_rows = []
    dataset_rows = []
    for policy, alpha, source_config in expected_policy_keys():
        count = 0
        if (policy, alpha, source_config) == (
            "global_balanced_pool",
            0.02,
            "spectral+interval",
        ):
            count = 17
        global_rows.append(
            {
                "pool_policy": policy,
                "alpha": str(alpha),
                "source_config": source_config,
                "datasets": str(datasets),
                "total_normal_instances": str(trials),
                "total_normal_review_count": str(count),
                "resolution_blocked_datasets": "0",
            }
        )
        for index in range(datasets):
            n_train = trials // datasets + int(index < trials % datasets)
            dataset_rows.append(
                {
                    "dataset": f"dataset_{index}",
                    "pool_policy": policy,
                    "alpha": str(alpha),
                    "source_config": source_config,
                    "n_train": str(n_train),
                    "normal_review_count": str(count if index == 0 else 0),
                    "resolution_blocked": "0",
                }
            )
    return global_rows, dataset_rows


class PolicyCoreTests(unittest.TestCase):
    def test_expected_grid_is_exactly_sixty_rows(self):
        keys = expected_policy_keys()
        self.assertEqual(len(keys), 60)
        self.assertEqual(len(set(keys)), 60)
        self.assertEqual({key[0] for key in keys}, set(POLICIES))
        self.assertEqual({key[1] for key in keys}, set(ALPHAS))
        self.assertEqual({key[2] for key in keys}, set(SOURCE_CONFIGS))

    def test_simultaneous_wilson_matches_preregistered_formula(self):
        successes, trials, comparisons, confidence = 17, 3863, 60, 0.95
        z = NormalDist().inv_cdf(1 - (1 - confidence) / comparisons)
        p_hat = successes / trials
        denominator = 1 + z * z / trials
        expected = (
            p_hat
            + z * z / (2 * trials)
            + z
            * math.sqrt(
                (p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials
            )
        ) / denominator
        self.assertAlmostEqual(
            simultaneous_wilson_upper(successes, trials, comparisons, confidence),
            expected,
            places=15,
        )

    def test_selection_uses_only_eligible_train_risk_rows(self):
        global_rows, dataset_rows = make_rows()
        result = select_train_risk_policy(
            global_rows,
            dataset_rows,
            expected_dataset_count=2,
        )
        self.assertEqual(
            result["selected"]["policy_key"],
            ["global_balanced_pool", 0.02, "spectral+interval"],
        )
        self.assertLessEqual(
            result["selected"]["simultaneous_wilson_upper"], 0.01
        )
        self.assertEqual(result["selected"]["selection_input_scope"], "train_oob_only")

    def test_selection_is_invariant_to_input_row_order(self):
        global_rows, dataset_rows = make_rows()
        baseline = select_train_risk_policy(
            global_rows,
            dataset_rows,
            expected_dataset_count=2,
        )
        random.Random(7).shuffle(global_rows)
        random.Random(11).shuffle(dataset_rows)
        shuffled = select_train_risk_policy(
            global_rows,
            dataset_rows,
            expected_dataset_count=2,
        )
        self.assertEqual(baseline, shuffled)

    def test_missing_grid_row_is_rejected(self):
        global_rows, dataset_rows = make_rows()
        with self.assertRaisesRegex(ValueError, "60-row policy grid"):
            select_train_risk_policy(
                global_rows[:-1],
                dataset_rows,
                expected_dataset_count=2,
            )

    def test_no_eligible_configuration_returns_abstention(self):
        global_rows, dataset_rows = make_rows()
        result = select_train_risk_policy(
            global_rows,
            dataset_rows,
            risk_target=1e-9,
            expected_dataset_count=2,
        )
        self.assertIsNone(result["selected"])
        self.assertEqual(result["status"], "policy_abstention")


if __name__ == "__main__":
    unittest.main()
