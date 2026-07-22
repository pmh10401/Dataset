from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import iisignature
import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery import (
    signature_source as signature_module,
)
from experiments.exp267_273_zero_support_fn_recovery.signature_source import (
    OrderedPathSignatureDeviation,
    signature_features,
)


class OrderedPathSignatureSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(269)
        self.rows = generator.normal(size=(8, 64)).astype(np.float32)

    def test_signature_dimension_matches_library_for_a_batch(self) -> None:
        features = signature_features(np.ones((4, 64), dtype=np.float32))
        expected_per_window = iisignature.siglength(3, 3)

        self.assertEqual(features.shape, (4, expected_per_window * 3))
        self.assertTrue(np.isfinite(features).all())

    def test_reversing_trace_changes_ordered_signature(self) -> None:
        forward = signature_features(self.rows)
        backward = signature_features(self.rows[:, ::-1])

        self.assertFalse(np.allclose(forward, backward))

    def test_signature_features_are_deterministic(self) -> None:
        np.testing.assert_array_equal(
            signature_features(self.rows), signature_features(self.rows)
        )

    def test_score_records_query_excluded_oof_and_label_free_provenance(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows,
            test_values=self.rows[:3] + 0.5,
            target_length=64,
        )

        result = OrderedPathSignatureDeviation().score(bundle, 17, Path("."))

        self.assertFalse(result.abstained)
        self.assertEqual(result.train_oof.shape, (8,))
        self.assertEqual(result.test_scores.shape, (3,))
        self.assertTrue(np.isfinite(result.train_oof).all())
        self.assertTrue(np.isfinite(result.test_scores).all())
        self.assertTrue(result.provenance["cross_fitted"])
        self.assertEqual(result.provenance["self_reference_violations"], 0)
        self.assertEqual(result.provenance["minimum_train"], 5)
        self.assertFalse(result.provenance["contains_test_labels"])
        self.assertEqual(result.provenance["signature"]["depth"], 3)
        self.assertEqual(result.provenance["signature"]["channels"], 3)
        self.assertEqual(
            result.provenance["signature"]["channel_order"],
            ("value", "first_difference", "normalized_time"),
        )
        self.assertEqual(
            result.provenance["signature"]["window_definitions"],
            (
                ("full", 0.0, 1.0),
                ("first_half", 0.0, 0.5),
                ("second_half", 0.5, 1.0),
            ),
        )
        self.assertEqual(
            result.provenance["signature_feature_dimension"],
            iisignature.siglength(3, 3) * 3,
        )
        self.assertEqual(
            result.provenance["signature_feature_dimension"],
            result.provenance["feature_dimension"],
        )
        self.assertEqual(result.provenance["iisignature_version"], "0.24")

    def test_score_abstains_below_minimum_train_without_using_labels(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows[:4],
            test_values=self.rows[:2],
            target_length=64,
        )

        result = OrderedPathSignatureDeviation().score(bundle, 17, Path("."))

        self.assertTrue(result.abstained)
        self.assertEqual(result.abstention_reason, "n_train_below_5")
        self.assertFalse(result.provenance["contains_test_labels"])

    def test_score_rejects_test_rows_not_matching_train_target_length(self) -> None:
        bundle = DatasetBundle(
            dataset="demo",
            train_normal=self.rows,
            test_values=self.rows[:2, :32],
            target_length=64,
        )

        with self.assertRaisesRegex(ValueError, "target length"):
            OrderedPathSignatureDeviation().score(bundle, 17, Path("."))

    def test_signature_features_reject_two_and_three_point_traces(self) -> None:
        for target_length in (2, 3):
            with self.subTest(target_length=target_length):
                with self.assertRaisesRegex(ValueError, "at least four"):
                    signature_features(np.ones((4, target_length), dtype=np.float32))

    def test_score_abstains_for_two_and_three_point_traces(self) -> None:
        for target_length in (2, 3):
            with self.subTest(target_length=target_length):
                rows = np.arange(8 * target_length, dtype=np.float32).reshape(
                    8, target_length
                )
                bundle = DatasetBundle(
                    dataset="short-demo",
                    train_normal=rows,
                    test_values=rows[:2],
                    target_length=target_length,
                )

                result = OrderedPathSignatureDeviation().score(bundle, 17, Path("."))

                self.assertTrue(result.abstained)
                self.assertFalse(result.eligible)
                self.assertEqual(result.abstention_reason, "target_length_below_4")
                self.assertEqual(result.train_oof.shape, (0,))
                self.assertEqual(result.test_scores.shape, (0,))
                self.assertFalse(result.provenance["contains_test_labels"])

    def test_four_point_trace_uses_only_nondegenerate_windows(self) -> None:
        rows = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
        original_signature = signature_module.iisignature.sig
        observed_shapes = []

        def record_signature(path: np.ndarray, depth: int) -> np.ndarray:
            observed_shapes.append(path.shape)
            return original_signature(path, depth)

        with patch.object(
            signature_module.iisignature, "sig", side_effect=record_signature
        ):
            features = signature_features(rows)

        self.assertEqual(
            observed_shapes,
            [(4, 4, 3), (4, 2, 3), (4, 2, 3)],
        )
        self.assertEqual(features.shape, (4, iisignature.siglength(3, 3) * 3))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
