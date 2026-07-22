from __future__ import annotations

import inspect
import unittest

from experiments.exp215_217 import core
from experiments.exp215_217.run_experiment_215_217 import summarize


class ShapeletStabilityStructureTests(unittest.TestCase):
    def test_shape_scoring_does_not_accept_test_labels(self) -> None:
        self.assertNotIn("y_test", inspect.signature(core.fit_shape_source).parameters)

    def test_fixed_dictionary_has_no_input_data_parameter(self) -> None:
        self.assertEqual(list(inspect.signature(core.fixed_kernel_dictionary).parameters), [])

    def test_exp218_candidate_rows_use_candidate_summary(self) -> None:
        rows = [{
            "variant": "bootstrap_consensus_medoid_pooled_cross_conformal__0.005",
            "candidate_tp": 2,
            "candidate_fp": 1,
        }]
        summary = summarize(rows, "218")
        self.assertEqual(summary[0]["candidate_count"], 3)


if __name__ == "__main__":
    unittest.main()
