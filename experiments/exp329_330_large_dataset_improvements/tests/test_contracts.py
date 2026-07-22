import inspect
import unittest

import numpy as np

from experiments.exp329_330_large_dataset_improvements.pipeline import (
    empirical_upper_pvalues,
    evidence_for_index,
    multiscale_features,
    predict_learned_dataset,
    select_verifier_variants,
)


class CandidateVerifierTests(unittest.TestCase):
    def test_evidence_counts_independent_source_families_and_rank(self):
        row = {
            "default_review_indices": [3],
            "primary_sources": {
                "rocket": {"candidate_indices": [3, 5], "order": [5, 3]},
                "frequency_image": {"candidate_indices": [5], "order": [3, 5]},
            },
            "independent_sources": {"change": [3, 5], "shape": [5]},
        }
        evidence = evidence_for_index(row, 5)
        self.assertEqual(evidence["primary_count"], 2)
        self.assertEqual(evidence["independent_count"], 2)
        self.assertEqual(evidence["total_support"], 4)
        self.assertEqual(evidence["best_primary_rank"], 1)
        self.assertFalse(evidence["in_existing_review"])

    def test_frozen_variants_do_not_promote_without_independent_support(self):
        evidence = {
            "in_existing_review": True,
            "primary_count": 3,
            "independent_count": 0,
            "total_support": 3,
            "best_primary_rank": 1,
        }
        selected = select_verifier_variants(evidence)
        self.assertFalse(any(selected.values()))

    def test_strict_outside_rule_requires_four_sources_and_two_independent(self):
        evidence = {
            "in_existing_review": False,
            "primary_count": 2,
            "independent_count": 2,
            "total_support": 4,
            "best_primary_rank": 9,
        }
        selected = select_verifier_variants(evidence)
        self.assertTrue(selected["outside_strict_4of8"])
        self.assertTrue(selected["combined_strict"])


class LearnedRepresentationTests(unittest.TestCase):
    def test_multiscale_features_have_fixed_finite_shape(self):
        rows = [np.linspace(0, 1, 19), np.sin(np.linspace(0, 3, 43))]
        features = multiscale_features(rows)
        self.assertEqual(features.shape, (2, 160))
        self.assertTrue(np.isfinite(features).all())

    def test_empirical_upper_pvalues_include_finite_sample_correction(self):
        calibration = np.asarray([1.0, 2.0, 3.0])
        query = np.asarray([0.0, 2.5, 4.0])
        np.testing.assert_allclose(empirical_upper_pvalues(calibration, query), [1.0, 0.5, 0.25])

    def test_prediction_entrypoint_has_no_test_label_argument(self):
        parameters = inspect.signature(predict_learned_dataset).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)


class RegistrationTests(unittest.TestCase):
    def test_sequential_runner_registers_both_experiments(self):
        import run_rank_experiments_sequential as runner

        self.assertEqual(runner.get_experiment("experiment_329_large_candidate_train_verifier")["expected_datasets"], 251)
        self.assertEqual(runner.get_experiment("experiment_330_large_learned_normal_representation")["expected_datasets"], 251)

    def test_dashboard_registers_both_experiments(self):
        import serve_rank_dashboard as dashboard

        self.assertIn("experiment_329_large_candidate_train_verifier", dashboard.EXPERIMENTS)
        self.assertIn("experiment_330_large_learned_normal_representation", dashboard.EXPERIMENTS)


if __name__ == "__main__":
    unittest.main()
