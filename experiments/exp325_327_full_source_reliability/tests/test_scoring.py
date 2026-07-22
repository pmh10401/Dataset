import unittest

from experiments.exp325_327_full_source_reliability.scoring import (
    adaptive_weights,
    choose_replacement,
    wilson_reliability,
)
from experiments.exp325_327_full_source_reliability.pipeline import _dataset_names


class FullSourceReliabilityTests(unittest.TestCase):
    def test_prior_reliability_weights_are_normalized(self):
        weights = adaptive_weights(
            {"rocket": 1.0, "frequency_image": 0.5, "texture_shape": 0.25,
             "aux_family_guard": 0.0, "aux_count_cap": 0.75},
            mode="prior_reliability",
        )
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights["rocket"], weights["aux_family_guard"])

    def test_pure_reliability_falls_back_to_fixed_priors(self):
        weights = adaptive_weights({name: 0.0 for name in (
            "rocket", "frequency_image", "texture_shape", "aux_family_guard", "aux_count_cap"
        )}, mode="pure_reliability")
        self.assertAlmostEqual(weights["rocket"], 0.36)

    def test_wilson_reliability_is_lower_for_more_exceedances(self):
        safe = wilson_reliability(0, 100)
        noisy = wilson_reliability(20, 100)
        self.assertGreater(safe["reliability"], noisy["reliability"])

    def test_replacement_contract_matches_exp93_guard(self):
        result = choose_replacement(
            base_indices={9}, weak_indices={9}, top_candidate=2,
            top_support=2, top_best_rank=3, score_gain=0.04,
            train_count=20,
        )
        self.assertEqual(result, {2})

    def test_tiny_train_never_replaces(self):
        result = choose_replacement(
            base_indices={9}, weak_indices={9}, top_candidate=2,
            top_support=5, top_best_rank=1, score_gain=1.0,
            train_count=10,
        )
        self.assertEqual(result, {9})

    def test_dataset_contract_is_exp291_lineage_not_all_database_rows(self):
        names = _dataset_names()
        self.assertEqual(len(names), 1117)
        self.assertNotIn("CornellWhaleChallenge", names)
        self.assertNotIn("Wafer_normal_1", names)


if __name__ == "__main__":
    unittest.main()
