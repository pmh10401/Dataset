from __future__ import annotations

import inspect
import json
import unittest

import numpy as np

from experiments.exp249_250_official_mini_multirocket.config import (
    ALPHAS,
    MODEL_CONFIGS,
    SEEDS,
)
from experiments.exp249_250_official_mini_multirocket.local_gap import (
    LocalGapModel,
)
from experiments.exp249_250_official_mini_multirocket.prediction import (
    build_prediction_row,
    predict_experiment,
    variants_from_archive,
)
from experiments.exp249_250_official_mini_multirocket.transforms import (
    expected_feature_dimension,
    make_transformer,
)


class OfficialRocketContractTest(unittest.TestCase):
    def test_preregistered_configuration_is_exact(self) -> None:
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05))
        self.assertEqual(SEEDS, (20260721, 20261721))
        self.assertEqual(MODEL_CONFIGS[249].requested_kernels, (1024, 4096))
        self.assertEqual(MODEL_CONFIGS[250].requested_kernels, (512, 1024))
        self.assertEqual(MODEL_CONFIGS[250].n_features_per_kernel, 4)
        self.assertFalse(MODEL_CONFIGS[250].normalise)

    def test_feature_dimensions_match_aeon_contract(self) -> None:
        self.assertEqual(expected_feature_dimension(MODEL_CONFIGS[249], 1024), 1008)
        self.assertEqual(expected_feature_dimension(MODEL_CONFIGS[249], 4096), 4032)
        self.assertEqual(expected_feature_dimension(MODEL_CONFIGS[250], 512), 4032)
        self.assertEqual(expected_feature_dimension(MODEL_CONFIGS[250], 1024), 8064)

    def test_transformers_are_official_aeon_estimators(self) -> None:
        mini = make_transformer(MODEL_CONFIGS[249], requested_kernels=1024, seed=SEEDS[0])
        multi = make_transformer(MODEL_CONFIGS[250], requested_kernels=512, seed=SEEDS[0])
        self.assertEqual(type(mini).__module__, "aeon.transformations.collection.convolution_based._minirocket")
        self.assertEqual(type(multi).__module__, "aeon.transformations.collection.convolution_based._multirocket")
        self.assertEqual(mini.n_jobs, 1)
        self.assertEqual(multi.n_jobs, 1)

    def test_local_gap_scores_far_queries_higher(self) -> None:
        reference = np.asarray([[0.0], [0.1], [-0.1], [0.05], [-0.05]])
        model = LocalGapModel(reference, neighbors=3)
        scores = model.score(np.asarray([[0.02], [8.0]]))
        self.assertGreater(scores[1], scores[0])

    def test_prediction_rows_exclude_evaluation_information(self) -> None:
        row = build_prediction_row(
            experiment=249,
            source="official_minirocket",
            variant="kernels_1024_seed_median",
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

    def test_prediction_function_does_not_accept_test_labels(self) -> None:
        signature = inspect.signature(predict_experiment)
        self.assertNotIn("y_test", signature.parameters)
        self.assertNotIn("labels", signature.parameters)

    def test_each_kernel_configuration_has_both_seed_variants(self) -> None:
        archive = {
            "kernels_1024__train_by_seed": np.ones((2, 5)),
            "kernels_1024__test_by_seed": np.ones((2, 3)),
            "kernels_4096__train_by_seed": np.ones((2, 5)),
            "kernels_4096__test_by_seed": np.ones((2, 3)),
        }
        self.assertEqual(
            set(variants_from_archive(249, archive)),
            {
                "kernels_1024_seed_median",
                "kernels_1024_seed_intersection",
                "kernels_4096_seed_median",
                "kernels_4096_seed_intersection",
            },
        )


if __name__ == "__main__":
    unittest.main()
