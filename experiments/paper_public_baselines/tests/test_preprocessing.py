from __future__ import annotations

import unittest

import numpy as np

from experiments.paper_public_baselines.contracts import DatasetValues
from experiments.paper_public_baselines.preprocessing import prepare_dataset


def _row(length: int, *, offset: float = 0.0) -> np.ndarray:
    return np.arange(length, dtype=np.float32) + np.float32(offset)


class PreprocessingTests(unittest.TestCase):
    def test_target_length_comes_only_from_train(self) -> None:
        raw = DatasetValues(
            dataset="toy",
            train_indices=(0, 1),
            train_rows=(_row(8), _row(10, offset=1.0)),
            test_indices=(0,),
            test_rows=(_row(200),),
        )

        prepared = prepare_dataset(raw)

        self.assertEqual(prepared.target_length, 9)
        self.assertEqual(prepared.train.shape, (2, 9))
        self.assertEqual(prepared.test.shape, (1, 9))
        self.assertTrue(np.isfinite(prepared.train).all())
        self.assertTrue(np.isfinite(prepared.test).all())

    def test_test_values_cannot_change_prepared_train(self) -> None:
        common = dict(
            dataset="toy",
            train_indices=(0, 1),
            train_rows=(_row(8), _row(10, offset=1.0)),
            test_indices=(0,),
        )
        short = prepare_dataset(DatasetValues(**common, test_rows=(_row(4),)))
        long = prepare_dataset(DatasetValues(**common, test_rows=(_row(400),)))

        self.assertEqual(short.target_length, long.target_length)
        np.testing.assert_array_equal(short.train, long.train)

    def test_nonfinite_and_constant_rows_become_finite(self) -> None:
        raw = DatasetValues(
            dataset="toy",
            train_indices=(0, 1),
            train_rows=(
                np.asarray([0.0, np.nan, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0]),
                np.ones(8, dtype=np.float32),
            ),
            test_indices=(0,),
            test_rows=(np.full(8, np.nan, dtype=np.float32),),
        )

        prepared = prepare_dataset(raw)

        self.assertTrue(np.isfinite(prepared.train).all())
        self.assertTrue(np.isfinite(prepared.test).all())
        np.testing.assert_array_equal(prepared.train[1], np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(prepared.test[0], np.zeros(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
