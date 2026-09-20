from __future__ import annotations

import ast
import inspect
from unittest import mock
import unittest

import numpy as np


class BlockedFoldsAntigravityTest(unittest.TestCase):
    def test_blocked_plan_matches_the_exact_1413_contiguous_ranges_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
            blocked_plan_antigravity,
        )

        folds = blocked_plan_antigravity(1413)
        self.assertEqual(len(folds), 5)
        self.assertEqual(
            [(int(fold.query_indices[0]), int(fold.query_indices[-1])) for fold in folds],
            [(0, 282), (283, 565), (566, 848), (849, 1130), (1131, 1412)],
        )
        self.assertEqual([len(fold.query_indices) for fold in folds], [283, 283, 283, 282, 282])
        self.assertEqual([len(fold.fit_indices) for fold in folds], [1130, 1130, 1130, 1131, 1131])
        self.assertEqual(
            np.bincount(np.concatenate([fold.query_indices for fold in folds]), minlength=1413).tolist(),
            [1] * 1413,
        )
        for fold_index, fold in enumerate(folds):
            with self.subTest(fold_index=fold_index):
                self.assertEqual(fold.n_rows, 1413)
                self.assertEqual(fold.fold_index, fold_index)
                self.assertEqual(np.intersect1d(fold.fit_indices, fold.query_indices).size, 0)
                self.assertEqual(
                    fold.fit_indices.tolist(),
                    [index for index in range(1413) if index not in set(fold.query_indices.tolist())],
                )
                self.assertFalse(np.shares_memory(fold.fit_indices, fold.query_indices))
                self.assertFalse(fold.fit_indices.flags.writeable)
                self.assertFalse(fold.query_indices.flags.writeable)

    def test_blocked_plan_preserves_array_split_behavior_for_other_valid_sizes_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
            blocked_plan_antigravity,
        )

        folds = blocked_plan_antigravity(7)
        self.assertEqual([fold.n_rows for fold in folds], [7, 7, 7, 7, 7])
        self.assertEqual([fold.query_indices.tolist() for fold in folds], [[0, 1], [2, 3], [4], [5], [6]])
        self.assertEqual([fold.fit_indices.tolist() for fold in folds], [[2, 3, 4, 5, 6], [0, 1, 4, 5, 6], [0, 1, 2, 3, 5, 6], [0, 1, 2, 3, 4, 6], [0, 1, 2, 3, 4, 5]])

    def test_blocked_plan_rejects_too_few_rows_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.blocked_folds_antigravity import (
            blocked_plan_antigravity,
        )

        for n_rows in (True, 0, 1, 2, 3, 4):
            with self.subTest(n_rows=n_rows), self.assertRaises(ValueError):
                blocked_plan_antigravity(n_rows)  # type: ignore[arg-type]

    def test_blocked_plan_never_calls_rng_or_shuffle_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import blocked_folds_antigravity

        with (
            mock.patch.object(
                blocked_folds_antigravity.np.random,
                "default_rng",
                side_effect=AssertionError("default_rng must not be used"),
            ),
            mock.patch.object(
                blocked_folds_antigravity.np.random,
                "shuffle",
                side_effect=AssertionError("shuffle must not be used"),
            ),
        ):
            blocked_folds_antigravity.blocked_plan_antigravity(1413)

    def test_blocked_plan_source_uses_array_split_without_random_calls_antigravity(self) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity import blocked_folds_antigravity

        tree = ast.parse(inspect.getsource(blocked_folds_antigravity))
        calls = {
            ast.unparse(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        self.assertIn("np.arange", calls)
        self.assertIn("np.array_split", calls)
        self.assertNotIn("np.random.default_rng", calls)
        self.assertNotIn("np.random.shuffle", calls)


if __name__ == "__main__":
    unittest.main()
