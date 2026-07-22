from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from experiments.exp251_sktime_statistical_features.independent_reaggregate import compare_summary
from experiments.exp251_sktime_statistical_features.prediction import (
    build_prediction_row,
    predict_experiment,
    variant_scores_from_archive,
)


class StatisticalPipelineContractTest(unittest.TestCase):
    def test_prediction_row_contains_no_evaluation_information(self) -> None:
        row = build_prediction_row(
            variant="combined_40",
            dataset="Example",
            alpha=0.01,
            n_test=4,
            candidates={1, 2},
            hard={1},
            standard={2},
            priority=set(),
            abstained=False,
            resolution_blocked=False,
            attainable_p=0.01,
        )
        forbidden = {"y_test", "labels", "tp", "fp", "f1", "combined_f1"}
        self.assertFalse(forbidden & set(row))
        json.dumps(row)

    def test_prediction_function_does_not_accept_labels(self) -> None:
        signature = inspect.signature(predict_experiment)
        self.assertNotIn("y_test", signature.parameters)
        self.assertNotIn("labels", signature.parameters)

    def test_archive_exposes_all_five_variants(self) -> None:
        archive = {
            f"{name}__train_oof": np.ones(5)
            for name in ("catch22_22", "summary_9", "hurst_3", "signature_6", "combined_40")
        }
        archive.update(
            {
                f"{name}__test_scores": np.ones(3)
                for name in ("catch22_22", "summary_9", "hurst_3", "signature_6", "combined_40")
            }
        )
        self.assertEqual(
            set(variant_scores_from_archive(archive)),
            {"catch22_22", "summary_9", "hurst_3", "signature_6", "combined_40"},
        )

    def test_independent_reaggregation_detects_tampering(self) -> None:
        results = [{"variant": "v", "alpha": "0.01", "candidate_tp": "2", "candidate_fp": "1", "candidate_f1": "0.8"}]
        summaries = [{"variant": "v", "alpha": "0.01", "candidate_count": "3", "candidate_tp": "2", "candidate_fp": "1", "candidate_precision": str(2 / 3), "candidate_mean_f1": "0.8"}]
        self.assertEqual(compare_summary(results, summaries, views=("candidate",)), [])
        summaries[0]["candidate_fp"] = "9"
        self.assertTrue(compare_summary(results, summaries, views=("candidate",)))


if __name__ == "__main__":
    unittest.main()
