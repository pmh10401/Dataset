import unittest

import numpy as np

from experiments.exp449_residual_fn_method_qualification_antigravity.contracts_antigravity import (
    MethodCandidateAntigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import (
    score_candidate_antigravity,
)
from services.time_series_anomaly_diagnosis.score_sources import (
    METHOD_NAMES,
    ScoreResult,
    crossfit_plan,
    score_method,
)


class ScoreSourcesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = np.asarray(
            [
                [float((row * 3 + column * column) % 23) for column in range(16)]
                for row in range(6)
            ],
            dtype=np.float32,
        )
        cls.test = np.asarray(
            [
                [float((row * 5 + column * 2) % 19) for column in range(16)]
                for row in range(3)
            ],
            dtype=np.float32,
        )

    def test_crossfit_plan_covers_each_train_row_once_in_five_deterministic_folds(self) -> None:
        folds = crossfit_plan(11, seed=20260721)

        self.assertEqual(len(folds), 5)
        self.assertEqual(
            tuple(query.tolist() for _, query in folds),
            ([0, 2, 8], [3, 9], [1, 5], [4, 6], [7, 10]),
        )
        self.assertEqual(
            sorted(np.concatenate([query for _, query in folds]).tolist()),
            list(range(11)),
        )

    def test_all_approved_methods_match_verified_exp449_for_both_seeds(self) -> None:
        for method_name in METHOD_NAMES:
            for seed in (20260721, 20261721):
                with self.subTest(method_name=method_name, seed=seed):
                    old = score_candidate_antigravity(
                        self.train,
                        self.test,
                        MethodCandidateAntigravity(method_name, method_name, (seed,), (0.005,)),
                        seed=seed,
                    )
                    new = score_method(self.train, self.test, method_name, seed)

                    np.testing.assert_array_equal(new.train_oof, old.train_oof)
                    np.testing.assert_array_equal(new.test_scores, old.test_scores)
                    self.assertEqual(new.feature_dimension, old.feature_dimension)
                    self.assertEqual(new.seed, seed)

    def test_score_rejects_unknown_hybrid_and_invalid_rows(self) -> None:
        for method_name in ("unknown", "hydra_multirocket_local_gap"):
            with self.subTest(method_name=method_name), self.assertRaises(ValueError):
                score_method(self.train, self.test, method_name, 20260721)
        with self.assertRaisesRegex(ValueError, "five TRAIN"):
            score_method(self.train[:4], self.test, "minirocket_local_gap", 20260721)
        with self.assertRaisesRegex(ValueError, "TEST"):
            score_method(self.train, self.test[:0], "minirocket_local_gap", 20260721)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            score_method(
                np.empty((5, 0)),
                np.empty((1, 0)),
                "minirocket_local_gap",
                20260721,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = self.train.copy()
            bad[0, 0] = np.nan
            score_method(bad, self.test, "minirocket_local_gap", 20260721)
        with self.assertRaisesRegex(ValueError, "shape"):
            score_method(self.train, self.test[:, :-1], "minirocket_local_gap", 20260721)
        with self.assertRaisesRegex(ValueError, "seed"):
            score_method(self.train, self.test, "minirocket_local_gap", True)

    def test_score_result_requires_exact_fields_and_protects_arrays_from_mutation(self) -> None:
        train = np.arange(5, dtype=float)
        test = np.arange(2, dtype=float)
        result = ScoreResult(train, test, 7, 1)
        train[0] = 99.0
        test[0] = 99.0

        self.assertEqual(result.train_oof[0], 0.0)
        self.assertEqual(result.test_scores[0], 0.0)
        with self.assertRaises(ValueError):
            result.train_oof.flags.writeable = True
        with self.assertRaises(ValueError):
            ScoreResult(np.array([np.nan]), np.array([0.0]), 1, 0)
        with self.assertRaises(ValueError):
            ScoreResult(np.array([0.0]), np.array([0.0]), True, 0)
        with self.assertRaises(ValueError):
            ScoreResult(np.array([0.0]), np.array([0.0]), 1, -1)


if __name__ == "__main__":
    unittest.main()
