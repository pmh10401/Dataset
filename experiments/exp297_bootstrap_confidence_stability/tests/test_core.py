from __future__ import annotations

import unittest

import numpy as np

from experiments.exp293_294_train_only_moe_lite.core import _loo_upper_pvalues
from experiments.exp297_bootstrap_confidence_stability.core import (
    bootstrap_alert_stability,
    deterministic_bootstrap_indices,
    fast_loo_upper_pvalues,
)


class BootstrapCoreTests(unittest.TestCase):
    def test_fast_loo_pvalues_match_existing_definition_with_ties(self) -> None:
        values = np.asarray([1.0, 4.0, 4.0, 2.0, 8.0])
        np.testing.assert_allclose(
            fast_loo_upper_pvalues(values),
            _loo_upper_pvalues(values),
        )

    def test_bootstrap_indices_are_deterministic_by_dataset_and_replicate(self) -> None:
        first = deterministic_bootstrap_indices("demo", replicate=3, n_train=20, seed=20260722)
        second = deterministic_bootstrap_indices("demo", replicate=3, n_train=20, seed=20260722)
        other = deterministic_bootstrap_indices("demo", replicate=4, n_train=20, seed=20260722)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))
        self.assertEqual(len(first), 20)

    def test_empty_support_is_stably_no_selected_support(self) -> None:
        train = {
            expert: {
                "v1": np.asarray([0.1, 0.2, 0.3, 0.4, 0.5]),
                "v2": np.asarray([0.2, 0.1, 0.4, 0.3, 0.6]),
            }
            for expert in ("statistical", "structural", "change_point", "official_multirocket")
        }
        result = bootstrap_alert_stability(
            dataset="demo",
            baseline_tiers={7: "no_selected_support"},
            supporting_experts_by_index={7: set()},
            train_scores_by_expert=train,
            replicates=8,
            seed=20260722,
            alpha=0.005,
        )

        self.assertTrue(result[7]["bootstrap_eligible"])
        self.assertEqual(result[7]["tier_match_count"], 8)
        self.assertEqual(result[7]["stability_rate"], 1.0)
        self.assertEqual(result[7]["bootstrap_tier_counts"], {"no_selected_support": 8})

    def test_train_below_five_abstains_without_numeric_stability(self) -> None:
        train = {
            expert: {"v1": np.asarray([0.1, 0.2, 0.3, 0.4])}
            for expert in ("statistical", "structural", "change_point", "official_multirocket")
        }
        result = bootstrap_alert_stability(
            dataset="small",
            baseline_tiers={2: "no_selected_support"},
            supporting_experts_by_index={2: set()},
            train_scores_by_expert=train,
            replicates=8,
            seed=20260722,
            alpha=0.005,
        )

        self.assertFalse(result[2]["bootstrap_eligible"])
        self.assertIsNone(result[2]["stability_rate"])
        self.assertEqual(result[2]["bootstrap_tier_counts"], {})


if __name__ == "__main__":
    unittest.main()
