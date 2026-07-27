import unittest

import numpy as np

from experiments.exp393_396.contracts import GrammarConfig
from experiments.exp393_396.grammar_core import (
    candidate_indices,
    conformal_p_value,
    crossfit_train_scores,
    fit_discrete_grammar,
    fit_grammar_model,
    score_state_path,
)


class GrammarScoreTests(unittest.TestCase):
    def test_unseen_transition_scores_above_normal_path(self):
        train_paths = [
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 2, 0],
        ]
        model = fit_discrete_grammar(train_paths, state_count=3)

        normal = score_state_path(model, [0, 1, 2, 0])
        abnormal = score_state_path(model, [0, 2, 1, 0])

        self.assertGreater(abnormal.transition, normal.transition)

    def test_long_dwell_scores_above_normal_dwell(self):
        model = fit_discrete_grammar(
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
            state_count=2,
        )

        normal = score_state_path(model, [0, 0, 1, 1])
        abnormal = score_state_path(model, [0, 0, 0, 0, 0, 1])

        self.assertGreater(abnormal.dwell, normal.dwell)

    def test_conformal_p_value_uses_finite_sample_correction(self):
        self.assertEqual(conformal_p_value([1.0, 2.0, 3.0], 4.0), 0.25)

    def test_candidate_indices_use_less_than_or_equal_alpha(self):
        self.assertEqual(candidate_indices([0.01, 0.02, 0.5], 0.02), [0, 1])

    def test_crossfit_excludes_held_out_trace(self):
        sequences = [
            np.column_stack(
                [
                    np.linspace(0.0, 1.0, 12) + index,
                    np.zeros((12, 5)),
                ]
            )
            for index in range(5)
        ]
        calls = []

        result = crossfit_train_scores(
            sequences,
            GrammarConfig(state_count=3),
            fit_observer=lambda indices: calls.append(tuple(indices)),
        )

        self.assertEqual(len(calls), 5)
        self.assertTrue(all(len(indices) == 4 for indices in calls))
        self.assertEqual(result["transition"].shape, (5,))
        self.assertEqual(result["composite"].shape, (5,))

    def test_crossfit_uses_at_most_five_deterministic_fits(self):
        sequences = [
            np.column_stack(
                [
                    np.linspace(0.0, 1.0, 12) + index,
                    np.zeros((12, 5)),
                ]
            )
            for index in range(10)
        ]
        calls = []

        crossfit_train_scores(
            sequences,
            GrammarConfig(state_count=3),
            fit_observer=lambda indices: calls.append(tuple(indices)),
        )

        self.assertEqual(len(calls), 5)
        self.assertTrue(all(len(indices) == 8 for indices in calls))
        excluded = [
            set(range(10)) - set(fit_indices)
            for fit_indices in calls
        ]
        self.assertEqual(set().union(*excluded), set(range(10)))

    def test_state_fitting_is_deterministic(self):
        sequences = [
            np.column_stack(
                [
                    np.sin(np.linspace(0.0, np.pi, 16) + offset),
                    np.cos(np.linspace(0.0, np.pi, 16) + offset),
                    np.zeros((16, 4)),
                ]
            )
            for offset in (0.0, 0.2, 0.4, 0.6, 0.8)
        ]
        config = GrammarConfig(state_count=4)

        first = fit_grammar_model(sequences, config)
        second = fit_grammar_model(sequences, config)

        np.testing.assert_allclose(
            first.clusterer.cluster_centers_,
            second.clusterer.cluster_centers_,
        )


if __name__ == "__main__":
    unittest.main()
