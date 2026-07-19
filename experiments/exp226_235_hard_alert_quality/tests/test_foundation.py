from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import DatasetBundle
from experiments.exp226_235_hard_alert_quality.database import load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import (
    align_bundle,
    sanitize_series,
    train_target_length,
)


class FoundationContractTests(unittest.TestCase):
    def test_prediction_bundle_has_no_label_field(self) -> None:
        self.assertNotIn("y_test", DatasetBundle.__dataclass_fields__)
        self.assertNotIn("labels", DatasetBundle.__dataclass_fields__)

    def test_values_loader_source_does_not_select_labels(self) -> None:
        source = inspect.getsource(load_values_bundle).lower()
        self.assertNotIn("select i.label", source)
        self.assertNotIn("select label", source)

    def test_target_length_uses_train_rows_only(self) -> None:
        target = train_target_length([np.arange(8), np.arange(10)])
        self.assertEqual(target, 9)

    def test_sanitize_interpolates_non_finite_values(self) -> None:
        actual = sanitize_series(np.asarray([1.0, np.nan, np.inf, 4.0]))
        np.testing.assert_allclose(actual, [1.0, 2.0, 3.0, 4.0])

    def test_align_bundle_uses_train_target_for_test_rows(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=(np.arange(8), np.arange(10)),
            test_values=(np.arange(20),),
        )
        aligned = align_bundle(bundle)
        self.assertEqual(aligned.train_normal.shape, (2, 9))
        self.assertEqual(aligned.test_values.shape, (1, 9))
        self.assertEqual(aligned.target_length, 9)


if __name__ == "__main__":
    unittest.main()
