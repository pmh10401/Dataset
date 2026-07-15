from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp166 import exp166_core as core


class Exp166CoreTests(unittest.TestCase):
    def test_all_frozen_sensitivity_rows_are_emitted(self):
        scores = {source: np.array([0.0, 1.0, 5.0]) for source in core.SOURCES}
        references = {
            method: {source: np.arange(200, dtype=float) / 100.0 for source in core.SOURCES}
            for method in core.METHODS
        }
        rows = core.candidate_rows(
            dataset="demo",
            n_train=8,
            n_test=3,
            normalized_test_scores=scores,
            references=references,
            selected_donors=["donor"],
            config_hash="abc",
        )
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["alpha"] for row in rows}, set(core.ALPHAS))
        self.assertEqual({row["agreement"] for row in rows}, {"2of3", "3of3"})
        self.assertEqual({row["method"] for row in rows}, set(core.METHODS))

    def test_prediction_rows_are_label_free(self):
        row = {
            "n_test": 2,
            "candidate_indices": [1],
            "candidate_count": 1,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        core.assert_label_free_prediction(row)
        row["f1"] = 1.0
        with self.assertRaises(ValueError):
            core.assert_label_free_prediction(row)

    def test_normalization_uses_train_reference(self):
        train = {source: np.array([0.0, 1.0, 2.0, 3.0]) for source in core.SOURCES}
        query = {source: np.array([1.5, 3.0]) for source in core.SOURCES}
        normalized_train, normalized_query = core.normalize_query_scores(train, query)
        for source in core.SOURCES:
            self.assertAlmostEqual(float(np.median(normalized_train[source])), 0.0)
            self.assertGreater(normalized_query[source][1], normalized_query[source][0])

    def test_evaluation_is_separate_and_exact(self):
        metrics = core.evaluate_indices([1, 2], np.array([0, 1, 0, 1]))
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 1))
        self.assertAlmostEqual(metrics["f1"], 0.5)

    def test_candidate_builder_has_no_label_argument(self):
        self.assertNotIn("y_test", inspect.signature(core.candidate_rows).parameters)
        self.assertNotIn("labels", inspect.signature(core.candidate_rows).parameters)


if __name__ == "__main__":
    unittest.main()
